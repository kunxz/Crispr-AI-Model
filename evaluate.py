# evaluate.py
import torch
import torch.nn as nn
import torch.nn.functional as Fjupy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in data_loader:
            outputs = model(sequences)
            all_predictions.extend(outputs.numpy())
            all_labels.extend(labels.numpy())

    # Compute evaluation metrics
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)

    return mse, mae, r2