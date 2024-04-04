import torch
from torch.utils.data import DataLoader


def predict(model: torch.nn.Module, loader: DataLoader, task: str = "binary"):
    """
    Performs inference using the given model and data loader, and returns the predictions along with
    the labels.
    This function supports both binary and multi-class (including regression) tasks by applying a
    sigmoid function to the output for binary tasks, and directly using the output for other tasks.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for data in loader:
        with torch.no_grad():

            y_pred = model(data)
            predicted_labels = torch.sigmoid(y_pred) if task == "binary" else y_pred

            all_preds.append(predicted_labels)
            if hasattr(data, "y"):
                y = data.y
            else:
                y = data["y"]
            all_labels.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_preds, all_labels
