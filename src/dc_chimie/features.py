"""Empreintes Morgan binaires, avec detection des SMILES invalides."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def morgan_fingerprints(smiles, radius=2, n_bits=2048, include_chirality=False):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=include_chirality
    )
    fingerprints = []
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) and value.strip() else None
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"SMILES invalide a la ligne {index}: {value!r}")
        fingerprints.append(generator.GetFingerprintAsNumPy(mol))
    return np.asarray(fingerprints, dtype=np.uint8).reshape(-1, n_bits)
