import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data.data import Data


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
