"""Local model registry shared by the API and command line."""
import json
from pathlib import Path

import joblib

from .dashboard_runtime import validate_queue
from .predict import load_bundle, predict_frame
from .nodal_predict import predict_frame as predict_nodal
from .recurrence_predict import predict_frame as predict_recurrence
from .reports import classification_runs, load_report


def extension_readiness(root="artifacts"):
    rows = []
    for task, directory in [("lymph_node_involvement", "nodal"), ("recurrence_competing_risk", "recurrence")]:
        root_path = Path(root)
        audit_path = root_path / f"{directory}_audit" / "audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
        selection = root_path / directory / "selection.json"
        rows.append({"task": task, "cohort_prepared": bool(audit),
                     "source_verified": bool(audit.get("training_allowed", False)),
                     "frozen_experiment_present": selection.is_file(),
                     "unresolved_requirements": audit.get("blocking_reason", "Prepare and audit the source cohort first."),
                     "state": "verification_required" if not audit.get("training_allowed") else
                              ("frozen_experiment_present" if selection.is_file() else "ready_for_training")})
    recorded = Path(root) / "recurrence_recorded" / "selection.json"
    if recorded.is_file():
        rows.append({"task": "recorded_recurrence", "cohort_prepared": True,
                     "source_verified": False, "frozen_experiment_present": True,
                     "state": "trained_exploratory",
                     "unresolved_requirements": "Independent validation and verified recurrence surveillance; no fixed time horizon.",
                     "prediction_target": "Source-recorded recurrence yes/no during variable follow-up"})
    return rows


def discover_models(root="artifacts"):
    registry = {}
    for run in classification_runs(root):
        report = load_report(run)
        for name in report["models"]:
            path = run / f"{name}.joblib"
            if path.is_file():
                registry[f"{run.name}/{name}"] = {"path": path, "task": report["task"],
                    "selected": name == report["selected_model_by_cv"], "run": run.name}
    for directory, task in [("nodal", "lymph_node_involvement"), ("recurrence", "recurrence_competing_risk"),
                            ("recurrence_recorded", "recorded_recurrence")]:
        run = Path(root) / directory
        if (run / "selection.json").is_file():
            selection = json.loads((run / "selection.json").read_text())
            for name in selection.get("frozen_models", {}):
                if name == "aalen_johansen_reference":
                    continue
                registry[f"{directory}/{name}"] = {"path": run / f"{name}.joblib", "task": task,
                                                  "selected": name == selection.get("selected_model_by_development"), "run": directory}
    return registry


def trusted_bundle(entry):
    path, task = entry["path"], entry["task"]
    if task in {"subtype", "malignancy"}:
        return load_bundle(path)
    selection = json.loads((path.parent / "selection.json").read_text())
    if task == "lymph_node_involvement":
        from .nodal_train import verify_frozen_models
        verify_frozen_models(path.parent, selection)
    elif task == "recurrence_competing_risk":
        from .recurrence_train import _verify_frozen_models
        _verify_frozen_models(path.parent, selection)
    elif task == "recorded_recurrence":
        from .recurrence_classifier import verify_recorded_models
        verify_recorded_models(path.parent, selection)
    else:
        raise ValueError("Unsupported model task.")
    return joblib.load(path)


def run_prediction(bundle, frame):
    validate_queue(bundle, frame)
    predictor = {"lymph_node_involvement": predict_nodal,
                 "recurrence_competing_risk": predict_recurrence}.get(bundle["task"], predict_frame)
    return predictor(bundle, frame)
