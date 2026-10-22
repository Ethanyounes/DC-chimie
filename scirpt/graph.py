import torch 
from rdkit import Chem 


def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-1, 0, 1]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def atome_features(atom):
    """Vecteur de features d'un atome : one-hot concaténés + deux booléens."""
    return (
        one_hot(atom.GetAtomicNum(), ATOM_TYPES)
        + one_hot(atom.GetDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_hot(atom.GetTotalNumHs(), NUM_HS)
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    )


BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def bond_features(bond):
    """Vecteur de features d'une liaison : type en one-hot + deux booléens."""
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    )