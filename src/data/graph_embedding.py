from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data.data import Data
from torch_geometric.datasets import MoleculeNet

from utils import set_random_seeds


def get_molecular_fingerprint_embedding(graph: Data, emb_dim: int) -> torch.Tensor:
    """
    Generates a molecular fingerprint embedding for a given molecule.

    This function converts a molecule represented by its SMILES string into a fixed-size
    fingerprint vector using the Morgan algorithm. The resulting fingerprint captures the
    presence of particular substructures within the molecule.
    """
    molecule = Chem.MolFromSmiles(graph.smiles[0])
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius=2, nBits=emb_dim
    )

    return torch.tensor(fingerprint).float()


if __name__ == "__main__":
    set_random_seeds(42)

    data_path = Path(__file__).parent.parent.parent / "data" / "Tox21"
    dataset = MoleculeNet(root=str(data_path), name="Tox21")[:1]
    for g in dataset:
        emb = get_molecular_fingerprint_embedding(g, emb_dim=64)
        emb_2 = get_molecular_fingerprint_embedding(g, emb_dim=64)
        print(f"emb = {emb}")
        for e in emb:
            print(f"e = {e}")
        assert torch.allclose(emb, emb_2)
