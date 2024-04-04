from data.data_loading import load_tox21_data_as_graphs, load_tox21_data_as_embeddings
from torch_geometric.loader import DataLoader as GeomDataLoader
from torch.utils.data import DataLoader


def test_load_tox21_data_as_graphs():
    dataloaders = load_tox21_data_as_graphs(binary_cls=True, upsampling=False)

    for data_loader in dataloaders:
        assert isinstance(data_loader, GeomDataLoader)

        for data in data_loader:
            assert hasattr(data, "x")
            assert hasattr(data, "y")
            assert hasattr(data, "edge_index")
            assert hasattr(data, "edge_attr")
            assert hasattr(data, "smiles")


def test_load_tox21_data_as_embeddings():
    emb_dim = 64
    dataloaders = load_tox21_data_as_embeddings(binary_cls=True, upsampling=False, emb_dim=emb_dim)

    for data_loader in dataloaders:
        assert isinstance(data_loader, DataLoader)

        for data in data_loader:
            assert data.keys().tolist() == ["x", "y"]
