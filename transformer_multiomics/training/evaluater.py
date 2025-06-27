from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import numpy as np
import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, device, is_transformer=False, model_name="Model"):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_dict, targets in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            targets = targets.to(device)
            x_dict = {k: v.to(device) for k, v in x_dict.items()}

            outputs = model(x_dict)
            if is_transformer:
                outputs = outputs[0]  # unpack predictions from (preds, attn_weights)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Combine all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute evaluation metrics
    r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
    evs = explained_variance_score(all_targets.flatten(), all_predictions.flatten())

    print(f"{model_name} Performance:")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "evs": evs,
        "predictions": all_predictions,
        "targets": all_targets,
    }
