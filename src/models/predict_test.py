import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from your_module import (
    predict,
)  # Assuming your function is in a module named "your_module"


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"x": torch.rand(2), "y": torch.randint(0, 2, ())}


def test_predict():
    # Dummy data
    model = nn.Linear(2, 1)
    dataset = DummyDataset(100)
    loader = DataLoader(dataset, batch_size=32)

    # Call the function
    all_preds, all_labels = predict(model, loader)

    # Check if the outputs are tensors
    assert isinstance(all_preds, torch.Tensor)
    assert isinstance(all_labels, torch.Tensor)

    # Check if the shapes are correct
    assert all_preds.shape[0] == all_labels.shape[0] == 100

    # Check if the task is binary
    assert torch.all(all_preds >= 0) and torch.all(all_preds <= 1)
