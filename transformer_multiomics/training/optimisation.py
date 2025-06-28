import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna

from transformer_multiomics.models.transformer import ModularMultiOmicsTransformer
from transformer_multiomics.data.data_preparation import prepare_data_loaders


def optimise_hyperparams(omics_set, n_trials=30, datasets=None, device=torch.device("cpu")):
    """
    Optimise hyperparameters for specific combination of omics data
    
    Parameters:
    -----------
    omics_set : list
        List of omics types to include
    n_trials : int
        Number of optimisation trials
        
    Returns:
    --------
    best_params : dict
        Dictionary of best hyperparameters
    """
    print(f"\n{'='*80}")
    print(f"Optimising hyperparameters for omics set: {omics_set}")
    print(f"{'='*80}")
    
    # Prepare data loaders with the selected omics types
    train_loader, val_loader, test_loader, input_dims, output_dim = prepare_data_loaders(omics_set)
    
    # Number of epochs for each trial
    epochs = 50
    
    # Define the Optuna objective function
    def objective(trial):
        # Suggest hyperparameters
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Additional hyperparameters for ModularMultiOmicsTransformer
        fusion_method = trial.suggest_categorical("fusion_method", 
                                                ["hierarchical", "late", "gated", "weighted", "cross_attention"])
        activation_function = trial.suggest_categorical("activation_function", 
                                                     ["gelu", "relu"])
        
        # Instantiate the model with the suggested hyperparameters
        model = ModularMultiOmicsTransformer(
            input_dims=input_dims,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            fusion_method=fusion_method,
            activation_function=activation_function
        ).to(device)
        
        # Define loss function and optimiser
        criterion = nn.MSELoss()
        optimiser = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Define the learning rate scheduler
        lr_scheduler = CosineAnnealingLR(
            optimiser,
            T_max=epochs,
            eta_min=1e-6
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience = 10
        counter = 0
        early_stop = False
        for epoch in range(epochs):
            if early_stop:
                break
                
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs_dict, targets in train_loader:
                inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
                targets = targets.to(device)
                
                optimiser.zero_grad()
                outputs = model(inputs_dict)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                
                running_loss += loss.item() * targets.size(0)
            
            lr_scheduler.step()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs_dict, targets in val_loader:
                    inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
                    targets = targets.to(device)
                    
                    outputs = model(inputs_dict)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * targets.size(0)
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            
            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                early_stop = True
        
        return best_val_loss

    # Run Optuna optimisation
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best hyperparameters
    best_params = study.best_params
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Check optimisation history 
    print(f"\nBest validation loss: {study.best_value:.6f}")
    
    return best_params
