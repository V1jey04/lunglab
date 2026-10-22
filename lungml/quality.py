"""Training-only input reference ranges; these flags are not clinical uncertainty."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import numeric_features
from .predict import load_bundle
from .recurrence_predict import read_feature_csv


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_path(model_path):
    return Path(model_path).with_suffix(".profile.json")


def create_profile(model_path, data_path, output=None):
    """Use only the fitted bundle's training patients, with no outcome columns."""
    model_path, data_path = Path(model_path), Path(data_path)
    output = Path(output) if output else profile_path(model_path)
    if output.resolve() in {model_path.resolve(), data_path.resolve()}:
        raise ValueError("Profile output must not overwrite the model or source data.")
    bundle = load_bundle(model_path)
    provenance = bundle.get("dataset_provenance", {})
    recorded_hashes = {provenance[k] for k in ("radiomics_dataset_sha256", "output_sha256") if provenance.get(k)}
    if recorded_hashes and sha256(data_path) not in recorded_hashes:
        raise ValueError("Source data does not match the dataset hashes saved with this fitted model.")
    ids = bundle.get("training_patient_ids", [])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("A unique training-patient list is required for a reference profile.")
    frame = read_feature_csv(data_path)
    if not set(ids).issubset(set(frame.patient_id)):
        raise ValueError("Source data does not contain every fitted training patient.")
    train = frame.set_index("patient_id").loc[ids]
    features = bundle["feature_names"]
    X = numeric_features(train, features)
    if X.isna().all().any():
        raise ValueError("A training feature has no observed values; cannot build its reference range.")
    stats = pd.DataFrame({"median": X.median(), "minimum": X.min(), "maximum": X.max(),
                          "q01": X.quantile(.01), "q99": X.quantile(.99)})
    result = {"schema_version": 1, "model_sha256": sha256(model_path),
              "source_sha256": sha256(data_path), "training_patients": len(ids),
              "source_hash_verified_against_bundle": bool(recorded_hashes),
              "training_patient_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              "feature_names": features, "statistics": stats.to_dict("index"),
              "method": "1st–99th percentile reference ranges from fitted training patients only",
              "review_fraction_threshold": .10,
              "limitation": "Heuristic input review. No calibrated uncertainty, site-drift guarantee or clinical acceptance threshold."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def load_profile(model_path, bundle=None):
    path = profile_path(model_path)
    if not path.is_file():
        return None
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != 1 or profile.get("model_sha256") != sha256(model_path):
        raise ValueError("Input reference profile does not match the saved model.")
    if bundle is not None and profile.get("feature_names") != bundle["feature_names"]:
        raise ValueError("Input reference profile has a different feature schema.")
    return profile


def assess_inputs(bundle, frame, profile=None):
    X = numeric_features(frame, bundle["feature_names"])
    if frame.empty:
        raise ValueError("No input cases.")
    rows = []
    if profile is not None:
        if profile["feature_names"] != bundle["feature_names"]:
            raise ValueError("Reference profile feature schema mismatch.")
        stats = pd.DataFrame.from_dict(profile["statistics"], orient="index").loc[X.columns]
        outside = X.lt(stats.q01) | X.gt(stats.q99)
    for position, (_, values) in enumerate(X.iterrows()):
        missing = values.index[values.isna()].tolist()
        unusual = outside.columns[outside.iloc[position]].tolist() if profile is not None else []
        fraction = len(unusual) / len(X.columns) if profile is not None else None
        rows.append({"patient_id": str(frame.iloc[position].get("patient_id", position)),
                     "missing_feature_count": len(missing), "missing_features": missing,
                     "reference_available": profile is not None,
                     "outside_reference_count": len(unusual) if profile is not None else None,
                     "outside_reference_fraction": fraction, "outside_reference_features": unusual,
                     "input_review_suggested": bool(missing or (fraction is not None and fraction > .10)),
                     "interpretation": "Training-range heuristic; not a validated uncertainty estimate."})
    return rows
