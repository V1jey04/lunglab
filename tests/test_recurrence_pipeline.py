import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from lungml import recurrence_train as trainer
from lungml.recurrence_predict import predict_frame
from lungml.reports import classification_runs, load_report


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def frozen_recurrence(tmp_path_factory):
    root = tmp_path_factory.mktemp("recurrence")
    rng = np.random.default_rng(18)
    n = 108
    kind = np.tile([0, 1, 2], 36)
    rows = pd.DataFrame({
        "patient_id": [f"{i:04d}" for i in range(n)],
        "sample_id": [f"SYN-{i:03d}" for i in range(n)],
        "time_days": np.where(kind == 0, rng.integers(1400, 2200, n), rng.integers(100, 1000, n)),
        "event_type": kind,
        "source_split": np.where(np.arange(n) < 90, "development", "test"),
        "eligible": True,
        "clinical_age_at_diagnosis": rng.normal(65, 7, n),
        "tumor_volume_source_units": rng.lognormal(3, 1, n),
    })
    for i in range(4):
        rows[f"original_firstorder_demo{i}"] = rng.normal(size=n)
    cohort = root / "cohort.csv"
    rows.to_csv(cohort, index=False)
    rows[["patient_id", "source_split"]].rename(columns={"source_split": "split"}).to_csv(
        cohort.with_suffix(".splits.csv"), index=False)
    cohort.with_suffix(".provenance.json").write_text(json.dumps({
        "output_sha256": sha256(cohort), "recurrence_free_followup_semantics_verified": True,
        "surgery_date_semantics_verified": True, "source_features_sha256": "synthetic",
        "source_clinical_sha256": "synthetic",
    }))
    audit = root / "audit.json"
    audit.write_text(json.dumps({"outputs": {"cohort": str(cohort)},
                                 "training_allowed": True, "synthetic": True}))
    output = root / "frozen"
    selection = trainer.run_recurrence(cohort, output, audit_path=audit)
    return cohort, output, rows, selection


def copy_run(frozen_recurrence, tmp_path):
    cohort, frozen, rows, selection = frozen_recurrence
    output = tmp_path / "run"
    shutil.copytree(frozen, output)
    return cohort, output, rows, selection


def test_frozen_evaluation_prediction_agreement_and_one_time_scoring(
        frozen_recurrence, tmp_path, monkeypatch):
    cohort, output, rows, selection = copy_run(frozen_recurrence, tmp_path)
    # A changed current configuration must not change the frozen evaluation.
    monkeypatch.setattr(trainer, "MINIMUM_CENSORING_SURVIVAL", 0.99)
    monkeypatch.setattr(trainer, "MINIMUM_AT_RISK", 10000)
    report = trainer.evaluate_recurrence(cohort, output)
    assert report["test_evaluated"]
    assert report["selected_model_by_development"] == selection["selected_model_by_development"]
    saved = pd.read_csv(output / "test_predictions.csv", dtype={"patient_id": str}).set_index("patient_id")
    test = rows.loc[rows.source_split.eq("test")].sample(frac=1, random_state=4)
    for family in ("clinical_cox", "radiomics_cox"):
        bundle = joblib.load(output / f"{family}.joblib")
        predicted = predict_frame(bundle, test)
        assert predicted.patient_id.str.len().eq(4).all()
        assert predicted.probability_sum_error.max() < 1e-12
        for horizon in (365, 730):
            estimates = predicted.loc[predicted.horizon_days_after_index.eq(horizon)].set_index("patient_id")
            np.testing.assert_allclose(
                estimates.recurrence_cumulative_incidence.sort_index(),
                saved[f"{family}_recurrence_cif_{horizon}d"].sort_index())
            assert report["test_metrics"][family][str(horizon)]["available"]
            assert report["paired_bootstrap_test_brier"][str(horizon)]["available"]
    before = {name: sha256(output / name) for name in ("external_report.json", "test_predictions.csv")}
    with pytest.raises(ValueError, match="already scored"):
        trainer.evaluate_recurrence(cohort, output)
    assert before == {name: sha256(output / name) for name in before}


@pytest.mark.parametrize("family", ["clinical_cox", "radiomics_cox", "aalen_johansen_reference"])
@pytest.mark.parametrize("change", ["modified", "missing", "missing_hash"])
def test_all_frozen_artifacts_checked_before_loading_or_scoring(
        frozen_recurrence, tmp_path, monkeypatch, family, change):
    cohort, output, _, _ = copy_run(frozen_recurrence, tmp_path)
    path = output / f"{family}.joblib"
    if change == "modified":
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "missing":
        path.unlink()
    else:
        selection_path = output / "selection.json"
        selection = json.loads(selection_path.read_text())
        del selection["frozen_models"][family]["model_sha256"]
        selection_path.write_text(json.dumps(selection))

    def forbidden(*args, **kwargs):
        pytest.fail("Loaded or scored a model before verifying all frozen artifacts")

    monkeypatch.setattr(trainer.joblib, "load", forbidden)
    monkeypatch.setattr(trainer, "predict_bundle", forbidden)
    with pytest.raises(ValueError, match=family):
        trainer.evaluate_recurrence(cohort, output)
    assert not (output / "external_report.json").exists()
    assert not (output / "test_predictions.csv").exists()


def test_replaced_valid_model_is_rejected(frozen_recurrence, tmp_path):
    cohort, output, _, selection = copy_run(frozen_recurrence, tmp_path)
    path = output / "clinical_cox.joblib"
    bundle = joblib.load(path)
    bundle["model"].recurrence_model_.coef_[0] += 2
    joblib.dump(bundle, path)
    assert sha256(path) != selection["frozen_models"]["clinical_cox"]["model_sha256"]
    with pytest.raises(ValueError, match="clinical_cox"):
        trainer.evaluate_recurrence(cohort, output)


def add_classification_report(root):
    run = root / "artifacts" / "subtype"
    run.mkdir(parents=True)
    labels = ["adenocarcinoma", "squamous_cell_carcinoma"]
    (run / "report.json").write_text(json.dumps({
        "task": "subtype", "data_provenance": "synthetic", "positive_class": labels[1],
        "selected_model_by_cv": "demo", "development_patients": 8, "test_patients": 4,
        "feature_names": ["original_firstorder_Mean"], "labels": labels,
        "models": {"demo": {"cv_selection_auc": 0.7,
                              "test_metrics": {"accuracy": 0.5, "confusion_matrix": [[1, 1], [1, 1]]}}},
        "interpretation": "Synthetic dashboard fixture.", "dataset_path": "missing.csv",
    }))
    Image.new("RGB", (2, 2)).save(run / "comparison.png")


@pytest.mark.parametrize("with_classification", [False, True])
def test_recurrence_dashboard_renders_with_and_without_classification(
        frozen_recurrence, tmp_path, monkeypatch, with_classification):
    cohort, frozen, _, _ = frozen_recurrence
    output = tmp_path / "artifacts" / "recurrence"
    shutil.copytree(frozen, output)
    trainer.evaluate_recurrence(cohort, output)
    if with_classification:
        add_classification_report(tmp_path)
    runs = classification_runs(tmp_path / "artifacts")
    assert [run.name for run in runs] == (["subtype"] if with_classification else [])
    with pytest.raises(ValueError, match="task-specific"):
        load_report(output)
    # Pandas Styler imports Matplotlib in the classification view. Warm its
    # fonts outside the timed UI run, with caches writable in sandboxed runs.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    import matplotlib.font_manager  # noqa: F401

    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=15)
    assert not app.exception
    assert "Recurrence" in [tab.label for tab in app.tabs]
    assert "Frozen recurrence experiment" in [heading.value for heading in app.subheader]
    selectors = [box for box in app.sidebar.selectbox if box.label == "Experiment"]
    assert len(selectors) == int(with_classification)
    if with_classification:
        assert selectors[0].options == ["subtype"]


def test_built_wheel_imports_recurrence_training_outside_checkout(tmp_path):
    pytest.importorskip("setuptools")
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy(PROJECT / "pyproject.toml", source / "pyproject.toml")
    shutil.copytree(PROJECT / "lungml", source / "lungml",
                    ignore=shutil.ignore_patterns("__pycache__"))
    wheels = tmp_path / "wheels"
    subprocess.run([sys.executable, "-c",
                    "from setuptools.build_meta import build_wheel; "
                    "import sys; build_wheel(sys.argv[1])", str(wheels)],
                   cwd=source, check=True, capture_output=True, text=True, timeout=30)
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next(wheels.glob("*.whl"))) as archive:
        assert "lungml/configs/recurrence.yaml" in archive.namelist()
        assert "lungml/assets/dashboard.css" in archive.namelist()
        archive.extractall(installed)
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import lungml
from lungml.recurrence_train import CONFIG, CONFIG_PATH
assert Path(lungml.__file__).is_relative_to(Path(sys.argv[1]))
assert CONFIG_PATH.is_file()
assert CONFIG['primary_horizon_days'] == 730
from lungml.cli import main
sys.argv = ['lung-ml', 'recurrence', 'train']
main()
"""
    result = subprocess.run([sys.executable, "-c", script, str(installed)],
                            cwd=installed, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "A recurrence audit is required" in result.stderr
    assert "FileNotFoundError" not in result.stderr
