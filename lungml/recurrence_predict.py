"""Inference helpers for the separate recurrence cumulative-incidence task."""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path

import joblib
import pandas as pd


def read_feature_csv(path) -> pd.DataFrame:
    """Read a feature CSV while rejecting ambiguous raw headers and IDs."""
    if hasattr(path, "getvalue"):
        payload = path.getvalue()
        text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
        stream = StringIO(text)
        source = BytesIO(text.encode("utf-8"))
    else:
        path = Path(path)
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
        stream = StringIO(text)
        source = BytesIO(payload)
    with stream:
        try:
            headers = [name.strip() for name in next(csv.reader(stream))]
        except StopIteration as error:
            raise ValueError("Feature CSV is empty.") from error
    if not headers or any(not name.strip() for name in headers):
        raise ValueError("Feature CSV must have nonempty column names.")
    duplicates = sorted({name for name in headers if headers.count(name) > 1})
    if duplicates:
        raise ValueError(f"Feature CSV has duplicate raw headers: {duplicates[:10]}")
    frame = pd.read_csv(source, header=0, names=headers,
                        dtype={"patient_id": "string", "sample_id": "string"})
    if "patient_id" not in frame:
        raise ValueError("Feature CSV must contain a patient_id column.")
    if frame.patient_id.isna().any() or frame.patient_id.str.strip().eq("").any():
        raise ValueError("patient_id values must be present and nonempty.")
    frame["patient_id"] = frame.patient_id.str.strip()
    if frame.patient_id.duplicated().any():
        raise ValueError("Feature CSV requires one row per unique patient_id.")
    return frame


def predict_frame(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    if bundle.get("task") != "recurrence_competing_risk" or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported recurrence model bundle.")
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError("Recurrence inference requires nonmissing patient_id values.")
    if frame.patient_id.astype(str).duplicated().any():
        raise ValueError("Recurrence inference requires unique patient_id values.")
    model = bundle.get("model")
    if model is None:
        raise ValueError("Recurrence bundle has no fitted model.")
    required = list(bundle.get("feature_names", ()))
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature CSV is missing model inputs: {missing[:10]}")

    curves = model.predict_curves(frame, bundle["horizons_days"])
    rows = []
    for index, patient_id in enumerate(frame.patient_id.astype(str)):
        missing_values = int(frame.iloc[index].loc[required].isna().sum())
        quality = "complete inputs" if missing_values == 0 else f"{missing_values} missing model inputs imputed"
        for horizon in bundle["horizons_days"]:
            prediction = curves["by_horizon"][int(horizon)]
            recurrence = float(prediction["recurrence_cif"][index])
            death = float(prediction["death_cif"][index])
            survival = float(prediction["event_free_survival"][index])
            rows.append({
                "patient_id": patient_id,
                "model_family": bundle["model_family"],
                "horizon_days_after_index": int(horizon),
                "recurrence_cumulative_incidence": recurrence,
                "competing_death_cumulative_incidence": death,
                "event_free_probability": survival,
                "probability_sum_error": abs(recurrence + death + survival - 1.0),
                "missing_model_input_count": missing_values,
                "input_quality": quality,
                "prediction_context": bundle.get("prediction_context", "research estimate"),
            })
    return pd.DataFrame(rows)


def predict_csv(model_path: str | Path, data_path: str | Path,
                output_path: str | Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    frame = read_feature_csv(data_path)
    result = predict_frame(bundle, frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
