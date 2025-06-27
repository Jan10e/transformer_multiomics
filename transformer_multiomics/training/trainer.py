import torch
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimiser,
    device,
    epochs=100,
    patience=15,
    model_path=None,
    return_attention=False
):
    best_loss = float("inf")
    counter = 0
    best_model_state = None

    train_losses = []
    test_losses = []
    all_attention_weights = [] if return_attention else None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x_dict, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            targets = targets.to(device)

            optimiser.zero_grad()
            outputs = model(x_dict)

            # Handle model output format: (prediction, attention) or just prediction
            if isinstance(outputs, tuple):
                predictions, _ = outputs
            else:
                predictions = outputs

            loss = criterion(predictions, targets)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * targets.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Evaluation
        model.eval()
        test_loss = 0.0
        if return_attention:
            all_attention_weights = []

        with torch.no_grad():
            for x_dict, targets in test_loader:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                targets = targets.to(device)

                outputs = model(x_dict)

                if isinstance(outputs, tuple):
                    predictions, attn_weights = outputs
                    if return_attention:
                        all_attention_weights.append(attn_weights.cpu())
                else:
                    predictions = outputs

                loss = criterion(predictions, targets)
                test_loss += loss.item() * targets.size(0)

        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        # Early stopping
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1

        # Progress logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Test Loss: {epoch_test_loss:.4f} | "
                f"Early Stopping: {counter}/{patience}"
            )

        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

    # Load best model and save
    model.load_state_dict(best_model_state)
    if model_path:
        torch.save(model.state_dict(), model_path)

    results = {
        "model": model,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "best_loss": best_loss,
    }

    if return_attention:
        results["attention_weights"] = all_attention_weights

    return results
