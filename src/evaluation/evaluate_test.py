import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.evaluate import auroc_score, bootstrap_auroc


def test_auroc_score():
    all_preds = torch.randint(0, 2, (100, 1))
    all_labels = torch.randint(0, 2, (100, 1))

    score = auroc_score(all_preds, all_labels)

    assert isinstance(score, torch.Tensor)
    assert score.shape == torch.Size([])
    assert 0 <= score <= 1


def test_auroc_score_equal_1():
    all_preds = torch.randint(0, 2, (100, 1))
    all_labels = all_preds

    score = auroc_score(all_preds, all_labels)

    assert score == 1.0


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.rand(2)


def test_bootstrap_auroc():
    model = nn.Linear(2, 2)
    dataset = DummyDataset(100)
    data_loader = DataLoader(dataset, batch_size=32)

    mean_score, std_score = bootstrap_auroc(model, data_loader)

    assert isinstance(mean_score, float)
    assert isinstance(std_score, float)
    assert 0 <= mean_score <= 1
