from pathlib import Path

import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import MoleculeNet

CARDIOTOXICITY = {
    "NR-PPAR-gamma": 6,
    "SR-ARE": 7,
    "SR-ATAD5": 8,
    "SR-p53": 11,
}
TOX_INDICES = list(CARDIOTOXICITY.values())


def set_binary_cardio_toxicity_labels(dataset: Dataset) -> None:
    """
    Converts multi-label toxicity indicators in a dataset to binary labels for cardiotoxicity.
    """
    y = dataset.data.y
    y_sum = torch.sum(y[:, TOX_INDICES], dim=1)
    y_binary = (y_sum > 0).float().unsqueeze(1)
    dataset.data.y = y_binary
    return None


if __name__ == "__main__":
    torch.manual_seed(42)

    data_path = Path(__file__).parent.parent.parent / "data" / "Tox21"
    dataset = MoleculeNet(root=str(data_path), name="Tox21")

    y = dataset.data.y
    set_binary_cardio_toxicity_labels(dataset)
    print(f"torch.sum(dataset.y) = {torch.sum(dataset.y)}")
    print(f"len(dataset.y) = {len(dataset.y)}")
    print(f"y.size() = {y.size()}")
