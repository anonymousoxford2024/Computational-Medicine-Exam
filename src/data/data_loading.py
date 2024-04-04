from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as GeomDataLoader, ImbalancedSampler

from data.data_exploration import set_binary_cardio_toxicity_labels
from data.graph_embedding import get_molecular_fingerprint_embedding


class EmbeddingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {"x": self.x[idx], "y": self.y[idx]}
        return sample


def shuffle(self, seed: int = 42):
    """
    Shuffles the given dataset using a specified random seed.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(self))
    dataset = self.index_select(perm)
    return dataset


def _get_train_val_test_dataloaders(
    ds: MoleculeNet,
    split: tuple = (0.7, 0.1, 0.2),
    batch_size: int = 32,
    upsampling: bool = False,
) -> Tuple[GeomDataLoader, GeomDataLoader, GeomDataLoader]:
    """
    Splits the dataset into training, validation, and test sets and returns their corresponding
    PyTorch Geometric dataloaders.
    """
    ds = shuffle(ds, seed=1)

    # Calculate split indices
    num_train = int(len(ds) * split[0])
    num_val = int(len(ds) * split[1])

    # Manually split the ds
    train_ds = ds[:num_train]
    val_ds = ds[num_train : num_train + num_val]
    test_ds = ds[num_train + num_val :]

    print(
        f"percentage of positive samples in train: {torch.sum(train_ds.y) / len(train_ds.y)}"
    )
    print(
        f"percentage of positive samples in val: {torch.sum(val_ds.y) / len(val_ds.y)}"
    )
    print(
        f"percentage of positive samples in test: {torch.sum(test_ds.y) / len(test_ds.y)}"
    )

    if upsampling:
        train_ds.y = train_ds.y.long()
        sampler = ImbalancedSampler(train_ds)
        train_dl = GeomDataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_dl = GeomDataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_dl = GeomDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = GeomDataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


def load_tox21_data_as_graphs(
    binary_cls: bool, upsampling: bool, batch_size: int = 32
) -> Tuple[GeomDataLoader, GeomDataLoader, GeomDataLoader]:
    """
    Loads the Tox21 dataset as graph data and prepares PyTorch Geometric dataloaders for
    training, validation, and testing.
    """
    data_path = Path(__file__).parent.parent.parent / "data" / "Tox21"
    dataset = MoleculeNet(root=data_path, name="Tox21")
    dataset.data.x = torch.nan_to_num(dataset.data.x)
    dataset.data.y = torch.nan_to_num(dataset.data.y)

    if binary_cls:
        set_binary_cardio_toxicity_labels(dataset)

    train_loader, val_loader, test_loader = _get_train_val_test_dataloaders(
        ds=dataset, batch_size=batch_size, upsampling=upsampling, split=(0.7, 0.1, 0.2)
    )
    return train_loader, val_loader, test_loader


def load_tox21_data_as_embeddings(
    binary_cls: bool, upsampling: bool, emb_dim: int, batch_size: int = 32
) -> Tuple[GeomDataLoader, GeomDataLoader, GeomDataLoader]:
    """
    Converts the Tox21 dataset from graph representations to molecular fingerprint embeddings and
    prepares dataloaders.
    """
    all_dataloader = load_tox21_data_as_graphs(
        binary_cls, upsampling=False, batch_size=1
    )
    new_dataloaders = []
    for dl in all_dataloader:
        embeddings_list = []
        labels_list = []

        for sample in dl:
            embeddings_list.append(
                get_molecular_fingerprint_embedding(sample, emb_dim=emb_dim)
            )
            labels_list.append(sample.y.squeeze().float())

        all_embeddings = torch.stack(embeddings_list)
        all_labels = torch.stack(labels_list).unsqueeze(dim=1)

        tensor_dataset = EmbeddingDataset(all_embeddings, all_labels)

        sampler = None
        if upsampling:
            class_sample_counts = torch.tensor(
                [(all_labels == t).sum() for t in torch.unique(all_labels, sorted=True)]
            )
            weight_per_class = 1.0 / class_sample_counts.float()
            weights = weight_per_class[all_labels.long()].double()
            sampler = WeightedRandomSampler(weights.squeeze(), len(weights))

        new_dl = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
        new_dataloaders.append(new_dl)

    return tuple(new_dataloaders)
