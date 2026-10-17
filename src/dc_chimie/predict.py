"""Generation d'une soumission depuis un modele sauvegarde."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import ROOT
from .features import morgan_fingerprints


def save_submission(model, X, ids, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": list(ids), "y": model.predict(X)})
    submission.to_csv(output, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--test", type=Path, default=ROOT / "data/raw/X_test_0mB4D2v.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    test = pd.read_csv(args.test)
    X = morgan_fingerprints(test["smiles"], **bundle["config"]["features"])
    print(save_submission(bundle["model"], X, test["id"], args.output))


if __name__ == "__main__":
    main()
