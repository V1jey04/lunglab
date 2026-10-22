"""Fixed TabICLv2 comparisons on development patients; no holdout reuse."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .data import TASK_LABELS, load_dataset
from .evaluation import metrics
from .improve import CorrelationPruner
from .multiview import CappedUnivariateSelector
from .train import patient_folds

CHECKPOINT = Path("data/raw/tabicl/tabicl-classifier-v2-20260212.ckpt")
CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
CHECKPOINT_URL = "https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt"


def foundation_model():
    from tabicl import TabICLClassifier
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("TabICLv2 checkpoint hash does not match the pinned release.")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("prune", CorrelationPruner()),
        ("select", CappedUnivariateSelector(k=50)),
        ("model", TabICLClassifier(n_estimators=8, batch_size=4,
                                 model_path=str(CHECKPOINT.resolve()),
                                 allow_auto_download=False, device="cpu", n_jobs=4,
                                 use_amp=False, random_state=42)),
    ])


def run(output="artifacts/foundation_development"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "development_report.json"
    if report_path.exists():
        raise ValueError("This development experiment has already run.")
    plan = {
        "evaluation": "5-fold stratified patient CV on development patients only",
        "seed": 42, "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fixed_parameters": {"n_estimators": 8, "features_maximum": 50,
                             "correlation_threshold": 0.95, "threshold": 0.5},
        "models": {}, "target_accuracy": 0.83, "target_met": False,
        "interpretation": "Exploratory development CV. Both cohort holdouts were previously inspected and are excluded. New independent patients are required for a final superiority claim.",
    }
    (output / "prespecified_plan.json").write_text(json.dumps(plan, indent=2))
    datasets = [
        ("lung1_original", "data/nsclc_subtype.csv", "artifacts/nsclc_lung1_converged/splits.csv", "original"),
        ("radiogenomics_original", "data/radiogenomics_subtype.csv", "data/radiogenomics_subtype.splits.csv", "original"),
        ("radiogenomics_rich", "data/radiogenomics_subtype.csv", "data/radiogenomics_subtype.splits.csv", "all"),
    ]
    for name, data, split_path, feature_mode in datasets:
        dataset = load_dataset(data, feature_mode=feature_mode)
        splits = pd.read_csv(split_path, dtype={"patient_id": str})
        development = set(splits.loc[splits.split == "development", "patient_id"])
        test = set(splits.loc[splits.split == "test", "patient_id"])
        use = np.isin(dataset.groups, list(development))
        X, y, groups = dataset.X.loc[use], dataset.y[use], dataset.groups[use]
        if set(groups) != development or set(groups) & test:
            raise ValueError("Preserved patient split does not match the data.")
        estimator = foundation_model()
        p = np.empty(len(y))
        folds = patient_folds(y, groups, 5, 42)
        print(f"TabICLv2 development: {name}, {len(y)} patients", flush=True)
        for fold, (tr, va) in enumerate(folds, 1):
            fitted = clone(estimator).fit(X.iloc[tr], y[tr])
            p[va] = fitted.predict_proba(X.iloc[va])[:, 1]
            m = metrics(y[va], p[va])
            print(f"  fold {fold}/5: accuracy {m['accuracy']:.3f}, balanced accuracy {m['balanced_accuracy']:.3f}", flush=True)
        result = {"metrics": metrics(y, p), "patients": len(y),
                  "majority_baseline_accuracy": float(max(np.bincount(y)) / len(y)),
                  "input_features": X.shape[1], "data": data,
                  "dataset_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                  "excluded_previously_inspected_test_ids": sorted(test)}
        plan["models"][name] = result
        pd.DataFrame({"patient_id": groups, "y_true": y, "probability": p,
                      "predicted": (p >= 0.5).astype(int)}).to_csv(output / f"{name}_development_oof.csv", index=False)
        estimator.fit(X, y)
        # Keep the pretrained checkpoint outside the serialized estimator: TabICL
        # restores it from the verified local path when the bundle is loaded.
        bundle = {"model": estimator, "feature_names": X.columns.tolist(),
                  "labels": list(TASK_LABELS["subtype"]), "task": "subtype",
                  "name": name, "threshold": 0.5, "data_provenance": "real",
                  "training_patient_ids": groups.tolist(), "evaluation": "development CV only",
                  "dataset_provenance": json.loads(Path(data).with_suffix(".provenance.json").read_text())}
        joblib.dump(bundle, output / f"{name}.joblib")
        report_path.write_text(json.dumps(plan, indent=2))
    plan["target_met"] = any(v["metrics"]["accuracy"] > 0.83 and
                              v["metrics"]["accuracy"] > v["majority_baseline_accuracy"]
                              for v in plan["models"].values())
    report_path.write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2), flush=True)
    return plan


if __name__ == "__main__":
    run()
