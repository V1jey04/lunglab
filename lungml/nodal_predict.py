"""Inference for the separate lymph-node involvement endpoint."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import numeric_features
from .nodal_data import LABELS
from .recurrence_predict import read_feature_csv


def validate_inputs(bundle, frame):
    if bundle.get("task") != "lymph_node_involvement" or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported nodal model bundle.")
    if not bundle.get("scan_timing_semantics_verified"):
        raise ValueError("Nodal bundle lacks verified preoperative scan timing.")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Provide nonempty nodal inputs with unique column names.")
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError("Nodal inference requires nonmissing patient_id values.")
    ids = frame.patient_id.astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Nodal inference requires one row per unique, nonempty patient_id.")
    names = bundle.get("feature_names", [])
    if not names or any(not name.startswith("original_") for name in names):
        raise ValueError("Nodal bundle has an invalid radiomics feature allowlist.")
    if tuple(bundle.get("labels", ())) != LABELS or bundle.get("model") is None:
        raise ValueError("Nodal bundle is missing its model or endpoint labels.")
    threshold = bundle.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Nodal threshold must be a probability.")
    X = numeric_features(frame, names)
    if X.isna().all(axis=1).any() or X.isna().mean(axis=1).gt(0.2).any():
        raise ValueError("More than 20% of required nodal inputs are missing in a case.")
    if "original_shape_MeshVolume" in X and (X.original_shape_MeshVolume < 0).any():
        raise ValueError("Tumour volume must be nonnegative.")
    return X


def predict_frame(bundle, frame):
    X = validate_inputs(bundle, frame)
    if not np.array_equal(bundle["model"].classes_, [0, 1]):
        raise ValueError("Nodal model classes do not match the frozen endpoint.")
    probabilities = np.asarray(bundle["model"].predict_proba(X))
    if (probabilities.shape != (len(frame), 2) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Nodal model returned invalid probabilities.")
    positive = probabilities[:, 1]
    result = pd.DataFrame({
        "patient_id": frame.patient_id.astype(str).str.strip().to_numpy(),
        "prediction": np.asarray(LABELS)[(positive >= bundle["threshold"]).astype(int)],
        "p_node_negative": probabilities[:, 0], "p_node_positive": positive,
        "decision_threshold": bundle["threshold"], "model_family": bundle["model_family"],
        "missing_model_input_count": X.isna().sum(axis=1).to_numpy(),
        "prediction_context": bundle["prediction_context"],
    })
    if "sample_id" in frame:
        result.insert(1, "sample_id", frame.sample_id.to_numpy())
    return result


def predict_csv(model_path, data_path, output_path="nodal_predictions.csv"):
    frame = read_feature_csv(data_path)
    result = predict_frame(joblib.load(model_path), frame)
    output_path = Path(output_path)
    if output_path.resolve() in {Path(model_path).resolve(), Path(data_path).resolve()}:
        raise ValueError("Nodal predictions must not overwrite the model or input CSV.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
