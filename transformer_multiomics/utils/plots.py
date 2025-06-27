import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, test_losses, early_stopping_epoch=None, title="Training and Validation Loss Curves"):
    """
    Plot training and validation loss curves.

    Args:
        train_losses (list): List of training losses.
        test_losses (list): List of validation/test losses.
        early_stopping_epoch (int, optional): Epoch where early stopping occurred. Will be marked with a vertical line if provided.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(len(train_losses)), train_losses, label="Training Loss", color="blue")
    ax.plot(range(len(test_losses)), test_losses, label="Validation Loss", color="red")

    if early_stopping_epoch is not None:
        ax.axvline(x=early_stopping_epoch, color="green", linestyle="--", label="Early Stopping Point")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_distribution(all_predictions, all_targets, model_name="Model"):
    """
    Plots a histogram of prediction errors.

    Parameters:
    - all_predictions (np.ndarray): Array of predicted values.
    - all_targets (np.ndarray): Array of ground truth values.
    - model_name (str): Optional title prefix (e.g., 'MLP', 'Transformer').
    """
    errors = (all_predictions - all_targets).flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, alpha=0.75)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{model_name} - Distribution of Prediction Errors")
    ax.grid(True, alpha=0.3)
    plt.show()
