"""Read original benchmarks and newer frozen-selection reports."""
import json
from pathlib import Path

from .data import TASK_LABELS


def report_file(run):
    run = Path(run)
    for name in ["external_report.json", "report.json"]:
        if (run / name).exists():
            return run / name
    raise FileNotFoundError(f"No completed evaluation in {run}")


def classification_runs(root="artifacts"):
    """Discover classification reports without routing other tasks to their UI."""
    candidates = {p.parent for pattern in ("*/report.json", "*/external_report.json")
                  for p in Path(root).glob(pattern)}
    runs = [run for run in candidates
            if json.loads(report_file(run).read_text()).get("task") in TASK_LABELS]
    return sorted(runs, key=lambda run: report_file(run).stat().st_mtime, reverse=True)


def load_report(run):
    report = json.loads(report_file(run).read_text())
    if "task" in report and report["task"] not in TASK_LABELS:
        raise ValueError(f"Use a task-specific view for report task: {report['task']}")
    report.setdefault("dataset_path", "data/nsclc_subtype.csv")
    if "test_metrics" in report:
        for name, result in report["models"].items():
            result["test_metrics"] = report["test_metrics"][name]
            result["cv_selection_auc"] = result["nested_cv_auc"]
            result["cv_balanced_accuracy"] = result["nested_cv_balanced_accuracy"]
            if "nested_cv_accuracy" in result:
                result["cv_accuracy"] = result["nested_cv_accuracy"]
            result.setdefault("prediction_rule", "native_predict")
            result["test_intervals"] = report.get("test_intervals", {}).get(name, {})
        report["selected_model_by_cv"] = report["selected_model_by_development"]
    return report
