"""Score a frozen classifier on a new cohort without refitting or retuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import metrics, bootstrap_intervals
from .predict import load_bundle
from .quality import assess_inputs, load_profile, sha256
from .recurrence_predict import read_feature_csv
from .service import run_prediction


def validate_external(model_path, data_path, output, source_evidence):
    model_path, data_path, output = Path(model_path), Path(data_path), Path(output)
    if not source_evidence or not source_evidence.strip():
        raise ValueError("Document independent-cohort identity and matching extraction in --source-evidence.")
    if output.exists():
        raise ValueError("Use a new validation directory; existing results cannot be overwritten.")
    bundle = load_bundle(model_path)
    if bundle["task"] not in {"subtype", "malignancy"}:
        raise ValueError("Use endpoint-specific evaluation for nodal or recurrence outcomes.")
    if not bundle.get("training_patient_ids"):
        raise ValueError("A fitted training-patient list is required to audit independence.")
    frame = read_feature_csv(data_path)
    known_ids = set(bundle["training_patient_ids"])
    splits = model_path.parent / "splits.csv"
    if splits.is_file():
        known_ids |= set(pd.read_csv(splits, dtype={"patient_id": str}).patient_id)
    if known_ids & set(frame.patient_id):
        raise ValueError("External cohort overlaps fitted or previously inspected experiment patients.")
    if "label" not in frame or set(frame.label) != set(bundle["labels"]):
        raise ValueError("The external cohort must contain both supported labels and no unknown labels.")
    counts = frame.label.value_counts()
    if counts.min() < 2:
        raise ValueError("At least two independent patients per class are required for intervals.")
    predictions = run_prediction(bundle, frame)
    y = frame.label.eq(bundle["labels"][1]).astype(int).to_numpy()
    probability = predictions[f"p_{bundle['labels'][1]}"].to_numpy()
    predicted = predictions.prediction.eq(bundle["labels"][1]).to_numpy()
    quality = assess_inputs(bundle, frame, load_profile(model_path, bundle))
    report = {"task": bundle["task"], "model_sha256": sha256(model_path),
              "cohort_sha256": sha256(data_path), "source_evidence": source_evidence,
              "patients": len(frame), "class_counts": counts.to_dict(), "frozen_threshold": bundle["threshold"],
              "metrics": metrics(y, probability, bundle["threshold"], predicted),
              "ci95": bootstrap_intervals(y, probability, threshold=bundle["threshold"], predicted=predicted),
              "input_review_patients": sum(q["input_review_suggested"] for q in quality),
              "independence_check": "No IDs overlap fitted patients or the local experiment split. Renamed patients and external provenance require source verification.",
              "interpretation": "No refitting, threshold tuning or model selection used this cohort. External source evidence is supplied by the researcher; results alone do not establish clinical utility."}
    output.mkdir(parents=True)
    predictions.assign(observed_label=frame.label.to_numpy()).to_csv(output / "predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False))
    return report
