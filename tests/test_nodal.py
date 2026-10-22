import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from lungml.nodal_data import prepare_nodal, load_nodal_cohort
from lungml.nodal_predict import predict_frame, predict_csv
from lungml import nodal_train as trainer
from lungml.dashboard_runtime import stream_inference
from lungml.recurrence_predict import read_feature_csv
from lungml.reports import classification_runs

PROJECT = Path(__file__).resolve().parents[1]


def source_tables(root, n=72):
    rng = np.random.default_rng(37)
    features = pd.DataFrame(rng.normal(size=(n, 10)),
                            columns=[f"original_firstorder_test{i}" for i in range(10)])
    features["original_shape_MeshVolume"] = rng.lognormal(3, 0.5, n)
    features["patient_id"] = [f"{i:04d}" for i in range(n)]
    features["label"] = np.tile(["adenocarcinoma", "squamous_cell_carcinoma"], n // 2)
    path = root / "source.csv"
    features.to_csv(path, index=False)
    pd.DataFrame({"patient_id": features.patient_id,
                  "split": np.where(np.arange(n) < n * 3 // 4, "development", "test")}).to_csv(
                      path.with_suffix(".splits.csv"), index=False)
    clinical = pd.DataFrame({
        "Case ID": features.patient_id,
        "Histology": np.tile(["Adenocarcinoma", "Squamous cell carcinoma"], n // 2),
        "Pathological N stage": np.tile(["N0", "N1", "N2"], n // 3),
        "CT Date": "2020-01-01", "Days between CT and surgery": 10,
        "Recurrence": "yes", "Date of Death": "2022-01-01", "Chemotherapy": "yes",
    })
    clinical_path = root / "clinical.csv"
    clinical.to_csv(clinical_path, index=False)
    return path, clinical_path, features, clinical


def prepare(root, *, verified=True):
    path, clinical, features, labels = source_tables(root)
    cohort, audit_dir = root / "nodal_cohort.csv", root / "audit"
    audit = prepare_nodal(path, clinical, cohort, audit_dir,
                          "Synthetic test fixture: matching CT precedes surgery by 10 days." if verified else None)
    return cohort, audit_dir / "audit.json", features, labels, audit


def test_unverified_timing_blocks_fitting_and_preserves_predictor_boundary(tmp_path):
    cohort, audit, features, _, result = prepare(tmp_path, verified=False)
    assert not result["training_allowed"]
    assert result["counts"]["source_node_positive"] == 48
    inputs = pd.read_csv(tmp_path / "nodal_features.csv")
    assert not {"label", "nodal_target", "Histology", "Pathological N stage", "Recurrence", "Chemotherapy"} & set(inputs)
    assert set(inputs) == {"patient_id", "sample_id", *[c for c in features if c.startswith("original_")]}
    with pytest.raises(ValueError, match="blocked"):
        load_nodal_cohort(cohort)
    with pytest.raises(ValueError, match="blocked"):
        trainer.run_nodal(cohort, tmp_path / "model", audit_path=audit)
    assert not (tmp_path / "model").exists()


def test_unknown_stage_and_invalid_timing_are_excluded_without_reassigning_split(tmp_path):
    path, clinical_path, _, clinical = source_tables(tmp_path)
    clinical.loc[0, "Pathological N stage"] = "NX"
    clinical.loc[1, "Days between CT and surgery"] = 100
    clinical.loc[2, "CT Date"] = "bad date"
    clinical.to_csv(clinical_path, index=False)
    audit = prepare_nodal(path, clinical_path, tmp_path / "cohort.csv", tmp_path / "audit")
    assert audit["counts"]["eligible_patients"] == 69
    rows = pd.read_csv(tmp_path / "cohort.csv", dtype={"patient_id": str})
    assert not {"0000", "0001", "0002"} & set(rows.patient_id)
    assert rows.source_split.eq("test").sum() == 18
    reasons = pd.read_csv(tmp_path / "audit" / "case_audit.csv", dtype={"patient_id": str})
    assert "unknown_pathological" in reasons.exclusion_reasons.iloc[0]


def test_nodal_rejects_duplicate_patients_and_output_source_collision(tmp_path):
    path, clinical_path, _, clinical = source_tables(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        prepare_nodal(path, clinical_path, path, tmp_path / "audit")
    clinical.loc[1, "Case ID"] = " 0000 "
    clinical.to_csv(clinical_path, index=False)
    with pytest.raises(ValueError, match="Unique"):
        prepare_nodal(path, clinical_path, tmp_path / "cohort.csv", tmp_path / "audit")


@pytest.fixture(scope="module")
def frozen_nodal(tmp_path_factory):
    root = tmp_path_factory.mktemp("nodal")
    cohort, audit, _, _, _ = prepare(root)
    output = root / "frozen"
    trainer.run_nodal(cohort, output, audit_path=audit)
    return cohort, audit, output


def test_frozen_prediction_evaluation_agreement_and_immutable_test_report(frozen_nodal, tmp_path):
    cohort, _, frozen = frozen_nodal
    output = tmp_path / "run"
    shutil.copytree(frozen, output)
    report = trainer.evaluate_nodal(cohort, output)
    assert report["test_evaluated"]
    rows, _ = load_nodal_cohort(cohort)
    test = rows.loc[rows.source_split.eq("test")].sample(frac=1, random_state=13)
    saved = pd.read_csv(output / "test_predictions.csv", dtype={"patient_id": str})
    for family in trainer.FAMILIES:
        bundle = joblib.load(output / f"{family}.joblib")
        assert not set(bundle["training_patient_ids"]) & set(test.patient_id)
        predicted = predict_frame(bundle, test).set_index("patient_id")
        observed = saved.loc[saved.model_family.eq(family)].set_index("patient_id")
        np.testing.assert_allclose(predicted.p_node_positive.sort_index(), observed.p_node_positive.sort_index())
        assert report["test_metrics"][family]["patients"] == len(test)
    original_hash = hashlib.sha256((output / "external_report.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already scored"):
        trainer.evaluate_nodal(cohort, output)
    assert hashlib.sha256((output / "external_report.json").read_bytes()).hexdigest() == original_hash
    with pytest.raises(ValueError, match="already frozen"):
        trainer.run_nodal(cohort, output, audit_path=frozen_nodal[1])


@pytest.mark.parametrize("family", trainer.FAMILIES)
def test_every_frozen_model_is_verified_before_loading(frozen_nodal, tmp_path, monkeypatch, family):
    cohort, _, frozen = frozen_nodal
    output = tmp_path / "run"
    shutil.copytree(frozen, output)
    path = output / f"{family}.joblib"
    path.write_bytes(path.read_bytes() + b"tampered")
    def forbidden(*args, **kwargs):
        pytest.fail("A model was loaded before all artifact hashes were checked")
    monkeypatch.setattr(trainer.joblib, "load", forbidden)
    with pytest.raises(ValueError, match=family):
        trainer.evaluate_nodal(cohort, output)
    assert not (output / "test_predictions.csv").exists()


def test_test_labels_do_not_change_development_selection(frozen_nodal, tmp_path):
    cohort, audit, frozen = frozen_nodal
    rows = pd.read_csv(cohort, dtype={"patient_id": str, "sample_id": str})
    rows.loc[rows.source_split.eq("test"), "nodal_target"] = 1 - rows.loc[rows.source_split.eq("test"), "nodal_target"]
    changed = tmp_path / "cohort.csv"
    rows.to_csv(changed, index=False)
    shutil.copyfile(cohort.with_suffix(".splits.csv"), changed.with_suffix(".splits.csv"))
    provenance = json.loads(cohort.with_suffix(".provenance.json").read_text())
    provenance["cohort_sha256"] = hashlib.sha256(changed.read_bytes()).hexdigest()
    changed.with_suffix(".provenance.json").write_text(json.dumps(provenance))
    manifest = json.loads(audit.read_text())
    manifest["outputs"]["cohort"] = str(changed)
    manifest["cohort_sha256"] = provenance["cohort_sha256"]
    changed_audit = tmp_path / "audit.json"
    changed_audit.write_text(json.dumps(manifest))
    second = trainer.run_nodal(changed, tmp_path / "changed", audit_path=changed_audit)
    first = json.loads((frozen / "selection.json").read_text())
    assert first["development_metrics"] == second["development_metrics"]
    assert first["selected_model_by_development"] == second["selected_model_by_development"]


def test_nodal_csv_and_streaming_reject_invalid_queue_before_prediction(frozen_nodal, tmp_path):
    cohort, _, frozen = frozen_nodal
    rows, _ = load_nodal_cohort(cohort)
    bundle = joblib.load(frozen / "radiomics_logistic.joblib")
    shuffled = rows.iloc[:4].sample(frac=1, random_state=2)
    source, destination = tmp_path / "input.csv", tmp_path / "predictions.csv"
    shuffled.to_csv(source, index=False)
    result = predict_csv(frozen / "radiomics_logistic.joblib", source, destination)
    assert result.patient_id.tolist() == shuffled.patient_id.tolist()
    assert result.patient_id.str.len().eq(4).all()
    events = list(stream_inference(bundle, shuffled))
    pd.testing.assert_frame_equal(events[-1].result, result)
    invalid = shuffled.copy()
    invalid.loc[invalid.index[-1], bundle["feature_names"]] = np.inf
    stream = stream_inference(bundle, invalid)
    with pytest.raises(ValueError, match="infinity"):
        next(stream)
    source.write_text("patient_id,original_a, original_a\n0001,1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_feature_csv(source)
    source.write_text("patient_id,original_a\n0001,1\n 0001 ,2\n")
    with pytest.raises(ValueError, match="unique"):
        read_feature_csv(source)


def test_both_extensions_render_without_subtype_models(frozen_nodal, tmp_path, monkeypatch):
    cohort, audit, frozen = frozen_nodal
    output = tmp_path / "artifacts" / "nodal"
    shutil.copytree(frozen, output)
    audit_dir = tmp_path / "artifacts" / "nodal_audit"
    audit_dir.mkdir()
    shutil.copyfile(audit, audit_dir / "audit.json")
    monkeypatch.chdir(tmp_path)
    assert classification_runs() == []
    app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=20)
    assert not app.exception
    assert {"Lymph-node spread", "Recurrence"} <= {tab.label for tab in app.tabs}
    assert "Frozen nodal experiment" in [heading.value for heading in app.subheader]
    app.checkbox(key="nodal_example").check().run(timeout=20)
    app.button(key="run_nodal").click().run(timeout=20)
    assert not app.exception and not app.error
    first = app.session_state["nodal_run"]
    assert len(first["result"]) == 5
    app.selectbox(key="nodal_patient").select(first["result"].patient_id.iloc[-1]).run(timeout=20)
    assert app.session_state["nodal_run"]["compute_ms"] == first["compute_ms"]
    app.checkbox(key="nodal_example").uncheck().run(timeout=20)
    assert "nodal_run" not in app.session_state


def test_cli_exposes_nodal_and_both_data_gates(tmp_path):
    for command in ["nodal", "recurrence"]:
        result = subprocess.run([sys.executable, "-m", "lungml.cli", command, "--help"],
                                cwd=PROJECT, text=True, capture_output=True, check=True)
        assert "prepare" in result.stdout and "predict" in result.stdout
    result = subprocess.run([sys.executable, "-m", "lungml.cli", "nodal", "train",
                             "--audit", str(tmp_path / "missing.json")],
                            cwd=PROJECT, text=True, capture_output=True)
    assert result.returncode != 0 and "audit is required" in result.stderr
