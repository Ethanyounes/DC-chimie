"""Validation, entrainement et suivi d'une experience reproductible."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

from .data import ROOT, load_data
from .features import morgan_fingerprints
from .predict import save_submission
from .validation import evaluate


def run(config_path=ROOT / "configs/baseline.yaml", output_dir=ROOT / "outputs"):
    config_path, output_dir = Path(config_path), Path(output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    train, test, y = load_data()
    print(f"{run_id}: {len(train)} molecules train, {len(test)} test", flush=True)
    X = morgan_fingerprints(train["smiles"], **config["features"])
    X_test = morgan_fingerprints(test["smiles"], **config["features"])
    model = RandomForestRegressor(random_state=config["seed"], **config["model"])
    print(f"Validation sur {config['folds']} folds...", flush=True)
    scores = evaluate(model, X, y, folds=config["folds"], seed=config["seed"])
    means = {metric: float(np.mean(values)) for metric, values in scores.items()}
    print(means, flush=True)
    model.fit(X, y)
    for folder in ("models", "submissions", "runs"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "models" / f"{run_id}.joblib"
    submission_path = output_dir / "submissions" / f"{run_id}.csv"
    joblib.dump({"model": model, "config": config, "run_id": run_id}, model_path)
    save_submission(model, X_test, test["id"], submission_path)
    metadata = {
        "run_id": run_id, "config": config, "scores_by_fold": scores, "means": means,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "rdkit", "scikit-learn", "joblib")},
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "data/raw").glob("*.csv"))},
        "model": str(model_path), "submission": str(submission_path),
    }
    (output_dir / "runs" / f"{run_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log = output_dir / "experiments.csv"
    row = {"run_id": run_id, "seed": config["seed"], "folds": config["folds"], **means,
           "config": json.dumps(config, sort_keys=True), "model": str(model_path), "submission": str(submission_path)}
    needs_header = not log.exists() or log.stat().st_size == 0
    with log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Soumission : {submission_path}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
