"""Measured inference events for the dashboard; no simulated model activity."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .data import numeric_features
from .predict import predict_frame
from .recurrence_predict import predict_frame as predict_recurrence
from .nodal_predict import predict_frame as predict_nodal, validate_inputs as validate_nodal


@dataclass
class InferenceEvent:
    stage: str
    completed: int
    total: int
    elapsed_ms: float
    compute_ms: float
    result: pd.DataFrame | None = None
    patient_id: str | None = None
    case_ms: float | None = None


def input_signature(model_path: Path, frame: pd.DataFrame) -> str:
    """Prevent results from being shown against a changed model or input queue."""
    digest = hashlib.sha256()
    digest.update(str(model_path.resolve()).encode())
    digest.update(str(model_path.stat().st_mtime_ns).encode())
    digest.update(json.dumps(frame.columns.tolist()).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def validate_queue(bundle: dict, frame: pd.DataFrame) -> None:
    if bundle.get("task") == "lymph_node_involvement":
        validate_nodal(bundle, frame)
        return
    if frame.empty:
        raise ValueError("Add at least one case to the queue.")
    if frame.columns.duplicated().any():
        raise ValueError("Input columns must have unique names.")
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError("Every case needs a patient_id.")
    ids = frame.patient_id.astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Queue one row per unique, nonempty patient_id.")
    features = list(bundle["feature_names"])
    X = numeric_features(frame, features)
    if bundle["task"] == "recurrence_competing_risk":
        volume = X["tumor_volume_source_units"]
        if volume.isna().any() or volume.lt(0).any():
            raise ValueError("Tumour volume must be finite and nonnegative.")
    else:
        if X.isna().all(axis=1).any():
            raise ValueError("A case has no model inputs.")
        if X.isna().mean(axis=1).gt(0.2).any():
            raise ValueError("More than 20% of required inputs are missing in a case.")
    if bundle["task"] == "recorded_recurrence":
        from .recurrence_classifier import validate_clinical_inputs
        validate_clinical_inputs(bundle, frame)


def stream_inference(bundle: dict, frame: pd.DataFrame):
    """Validate the whole queue, then yield results from actual case-level calls."""
    started = perf_counter()
    validate_queue(bundle, frame)
    total, compute_ms = len(frame), 0.0
    yield InferenceEvent("validated", 0, total, (perf_counter() - started) * 1000, compute_ms)
    predictor = {"recurrence_competing_risk": predict_recurrence,
                 "lymph_node_involvement": predict_nodal}.get(bundle["task"], predict_frame)
    outputs = []
    for position in range(total):
        row = frame.iloc[[position]]
        case_started = perf_counter()
        result = predictor(bundle, row)
        case_ms = (perf_counter() - case_started) * 1000
        compute_ms += case_ms
        outputs.append(result)
        yield InferenceEvent("prediction", position + 1, total,
                             (perf_counter() - started) * 1000, compute_ms, result,
                             str(row.patient_id.iloc[0]), case_ms)
    yield InferenceEvent("complete", total, total, (perf_counter() - started) * 1000,
                         compute_ms, pd.concat(outputs, ignore_index=True))


def preserved_cases(run: Path, report: dict) -> pd.DataFrame:
    """Join by string IDs; never send development rows into the example queue."""
    source = Path(report["dataset_path"])
    splits_path = run / "splits.csv"
    if not source.is_file() or not splits_path.is_file():
        return pd.DataFrame()
    source_frame = pd.read_csv(source, dtype={"patient_id": str, "sample_id": str})
    splits = pd.read_csv(splits_path, dtype={"patient_id": str})
    test_ids = set(splits.loc[splits.split.eq("test"), "patient_id"])
    return source_frame.loc[source_frame.patient_id.isin(test_ids)].reset_index(drop=True)


def model_comparison(report: dict) -> pd.DataFrame:
    rows = []
    for name, value in report["models"].items():
        test = value.get("test_metrics", {})
        rows.append({
            "Model": name,
            "Development AUC": value.get("cv_selection_auc"),
            "Development accuracy": value.get("cv_accuracy"),
            "Test accuracy": test.get("accuracy"),
            "Balanced accuracy": test.get("balanced_accuracy"),
            "Test AUC": test.get("roc_auc"),
            "Selected": name == report["selected_model_by_cv"],
        })
    return pd.DataFrame(rows)
