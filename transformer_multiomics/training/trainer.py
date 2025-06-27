import torch
from tqdm import tqdm
from pathlib import Path

def train_model(model, train_loader, test_loader, criterion, optimizer, device, 
                epochs=100, patience=15, model_name="model", save_path=None, 
                log_interval=10, return_attention_weights=False):
    """
    General training loop for PyTorch models with early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda/cpu)
        epochs: Maximum number of epochs
        patience: Early stopping patience
        model_name: Name for saving the model
        save_path: Path to save the best model (optional)
        log_interval: How often to print progress
        return_attention_weights: Whether model returns attention weights (for transformer)
        
    Returns:
        tuple: (trained_model, train_losses, test_losses, best_loss) or 
               (trained_model, train_losses, test_losses, best_loss, attention_weights) 
               if return_attention_weights=True
    """
    
    # Early stopping setup
    best_loss = float("inf")
    counter = 0
    best_model_state = None
    
    # Track losses
    train_losses = []
    test_losses = []
    
    print(f"Starting training for {model_name}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for x_dict, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Move data to device
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Handle models that return attention weights
            if return_attention_weights:
                outputs, _ = model(x_dict)
            else:
                outputs = model(x_dict)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        all_attention_weights = [] if return_attention_weights else None
        
        with torch.no_grad():
            for x_dict, targets in test_loader:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                targets = targets.to(device)
                
                # Forward pass
                if return_attention_weights:
                    outputs, attn_weights = model(x_dict)
                    all_attention_weights.append(attn_weights.cpu())
                else:
                    outputs = model(x_dict)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
        
        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        
        # Early stopping check
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
        
        # Progress logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | "
                  f"Test Loss: {epoch_test_loss:.4f} | Early Stopping: {counter}/{patience}")
        
        # Early stopping
        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # If we need attention weights, collect them from the best model
    final_attention_weights = None
    if return_attention_weights:
        print("Collecting attention weights from best model...")
        model.eval()
        final_attention_weights = []
        with torch.no_grad():
            for x_dict, targets in test_loader:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                targets = targets.to(device)
                
                outputs, attn_weights = model(x_dict)
                final_attention_weights.append(attn_weights.cpu())
    
    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / f"best_{model_name}_model.pt"
        torch.save(model.state_dict(), model_file)
        print(f"Best model saved to {model_file}")
    
    # Return attention weights if collected
    if return_attention_weights and final_attention_weights:
        return model, train_losses, test_losses, best_loss, final_attention_weights
    else:
        return model, train_losses, test_losses, best_loss