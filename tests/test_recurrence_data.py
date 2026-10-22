import numpy as np
import pandas as pd
import pytest

from lungml.recurrence_data import prepare_recurrence


@pytest.fixture
def source_tables(tmp_path):
    rng = np.random.default_rng(4)
    ids = [f"{i:03d}" for i in range(1, 7)]
    labels = ["adenocarcinoma", "squamous_cell_carcinoma"] * 3
    features = pd.DataFrame(rng.normal(size=(6, 10)),
                            columns=[f"original_firstorder_test{i}" for i in range(10)])
    features["original_shape_MeshVolume"] = np.arange(6) + 10
    features["patient_id"], features["label"] = ids, labels
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    pd.DataFrame({"patient_id": ids, "split": ["development"] * 4 + ["test"] * 2}).to_csv(
        feature_path.with_suffix(".splits.csv"), index=False)
    clinical = pd.DataFrame({
        "Case ID": ids,
        "Histology": ["Adenocarcinoma", "Squamous cell carcinoma"] * 3,
        "Pathological M stage": "M0", "CT Date": "2020-01-01",
        "Days between CT and surgery": 10, "Age at Histological Diagnosis": 65,
        "Recurrence": ["yes", "no", "no", "yes", "yes", "no"],
        "Survival Status": ["dead", "dead", "alive", "dead", "alive", "alive"],
        "Date of Recurrence": ["2020-03-01", None, None, "2020-03-01", "2020-02-01", None],
        "Date of Death": ["2020-04-01", "2020-05-01", None, "2020-03-01", None, None],
        "Date of Last Known Alive": "2020-03-01",
    })
    return feature_path, clinical


@pytest.mark.parametrize("column", ["Recurrence", "Survival Status", "Pathological M stage"])
@pytest.mark.parametrize("value", [None, "unknown"])
def test_missing_or_unknown_status_is_audited_and_excluded(tmp_path, source_tables, column, value):
    features, clinical = source_tables
    clinical.loc[2, column] = value
    clinical_path = tmp_path / "clinical.csv"
    clinical.to_csv(clinical_path, index=False)
    output, audit_dir = tmp_path / "cohort.csv", tmp_path / "audit"
    result = prepare_recurrence(features, clinical_path, output, audit_dir)
    audit = pd.read_csv(audit_dir / "case_audit.csv", dtype={"patient_id": str})
    excluded = audit.set_index("patient_id").loc["003"]
    assert not excluded.eligible_90_day_primary
    assert "missing_or_unknown" in excluded.exclusion_reasons
    assert "003" not in set(pd.read_csv(output, dtype={"patient_id": str}).patient_id)
    assert result["counts"]["eligible_by_dates_and_clinical_rules"] == 5
    assert not result["training_allowed"]


def test_both_missing_statuses_with_recorded_event_dates_do_not_interrupt_audit(tmp_path, source_tables):
    features, clinical = source_tables
    clinical.loc[0, ["Recurrence", "Survival Status"]] = None
    clinical_path = tmp_path / "clinical.csv"
    clinical.to_csv(clinical_path, index=False)
    prepare_recurrence(features, clinical_path, tmp_path / "cohort.csv", tmp_path / "audit")
    audit = pd.read_csv(tmp_path / "audit" / "case_audit.csv", dtype={"patient_id": str})
    excluded = audit.set_index("patient_id").loc["001"]
    assert not excluded.eligible_90_day_primary
    assert "missing_or_unknown_recurrence_status" in excluded.exclusion_reasons
    assert "missing_or_unknown_survival_status" in excluded.exclusion_reasons


def test_first_event_and_same_day_tie_rules_are_preserved(tmp_path, source_tables):
    features, clinical = source_tables
    clinical_path = tmp_path / "clinical.csv"
    clinical.to_csv(clinical_path, index=False)
    prepare_recurrence(features, clinical_path, tmp_path / "cohort.csv", tmp_path / "audit")
    outcomes = pd.read_csv(tmp_path / "recurrence_outcomes.csv", dtype={"patient_id": str})
    assert outcomes.event_type.tolist() == [1, 2, 0, 1, 1, 0]
    assert outcomes.eligible.all()
    assert outcomes.time_days.gt(0).all()
