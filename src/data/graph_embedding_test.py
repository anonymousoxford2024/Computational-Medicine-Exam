import torch
from torch_geometric.data.data import Data

from data.graph_embedding import get_molecular_fingerprint_embedding


def test_get_molecular_fingerprint_embedding():

    g = Data(smiles="CN1CCC[C@H]1c2cccnc2")
    emb = get_molecular_fingerprint_embedding(g, emb_dim=64)

    assert isinstance(emb, torch.Tensor)
    assert emb.size() == (64,)
