import torch
from torch_geometric.data import InMemoryDataset, Data

from data.data_exploration import set_binary_cardio_toxicity_labels


class DummyGraphDataset(InMemoryDataset):
    def __init__(self, root, size, num_nodes, y):
        self.size = size
        self.num_nodes = num_nodes
        super(DummyGraphDataset, self).__init__(root)
        self.data = Data(
            x=torch.rand(50, 10), edge_index=torch.randint(0, 50, size=(2, 100)), y=y
        )


def test_set_binary_cardio_toxicity_labels() -> None:
    y = torch.zeros((5, 12))
    y[0, 6] = 1.0
    y[2, 1] = 1.0
    y[2, 6] = 1.0
    y[3, 2] = 1.0
    y[3, 8] = 1.0
    y[3, 11] = 1.0
    y[4, 10] = 1.0
    dataset = DummyGraphDataset(
        root="/tmp/DummyGraphDataset", size=100, num_nodes=10, y=y
    )
    assert dataset.data.y.size() == (5, 12)

    set_binary_cardio_toxicity_labels(dataset)
    assert dataset.data.y.size() == (5, 1)
    assert torch.all(
        dataset.data.y == torch.tensor([[1.0], [0.0], [1.0], [1.0], [0.0]])
    )
