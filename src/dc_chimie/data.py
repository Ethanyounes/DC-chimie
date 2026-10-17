"""Chargement des donnees brutes et alignement explicite par identifiant."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_data(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data" / "raw"
    train = pd.read_csv(data_dir / "X_train_AX2CWD7.csv")
    test = pd.read_csv(data_dir / "X_test_0mB4D2v.csv")
    targets = pd.read_csv(data_dir / "y_train_hCIvDMj.csv")
    for name, frame, columns in (
        ("train", train, ["id", "smiles"]),
        ("test", test, ["id", "smiles"]),
        ("targets", targets, ["id", "y"]),
    ):
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"Colonnes manquantes dans {name}: {columns}")
        if frame[columns].isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"Valeurs manquantes ou identifiants dupliques dans {name}")
    if set(train["id"]) != set(targets["id"]):
        raise ValueError("Les identifiants train et targets ne correspondent pas")
    y = targets.set_index("id").loc[train["id"], "y"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Les cibles doivent etre finies")
    return train, test, y
