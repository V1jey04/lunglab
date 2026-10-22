"""Audited patient-level nodal labels, kept separate from model inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset, numeric_features
from .recurrence_data import _sha256, _validate_csv_headers, _label_matches

DEFAULT_COHORT = Path("data/nodal_cohort.csv")
DEFAULT_AUDIT = Path("artifacts/nodal_audit")
LABELS = ("node_negative", "node_positive")
TIMING_NOTE = (
    "Verify that the extracted radiomic CT matches the clinical scan and that the "
    "positive CT-to-surgery interval means CT preceded surgery."
)


def _patient_ids(frame, column):
    if column not in frame or frame[column].isna().any():
        raise ValueError(f"Unique, nonmissing {column} values are required.")
    values = frame[column].astype(str).str.strip()
    if values.eq("").any() or values.duplicated().any():
        raise ValueError(f"Unique, nonempty {column} values are required.")
    return values


def prepare_nodal(features_path="data/radiogenomics_subtype.csv",
                  clinical_path="data/raw/nsclc_radiogenomics/clinical.csv",
                  output=DEFAULT_COHORT, audit_dir=DEFAULT_AUDIT,
                  surgery_date_evidence=None):
    """Prepare N0 versus N1/N2/N3 without inventing clinical staging labels.

    Positive intervals of 1–90 days define a provisional preoperative cohort.
    Fitting remains blocked until a documented scan/date interpretation exists.
    Pathological stage, histology, treatment and recurrence are never predictors.
    """
    features_path, clinical_path = Path(features_path), Path(clinical_path)
    output, audit_dir = Path(output), Path(audit_dir)
    split_path = features_path.with_suffix(".splits.csv")
    destinations = {
        "cohort": output, "features": output.with_name("nodal_features.csv"),
        "outcomes": output.with_name("nodal_outcomes.csv"),
        "splits": output.with_suffix(".splits.csv"),
        "provenance": output.with_suffix(".provenance.json"),
        "case_audit": audit_dir / "case_audit.csv", "audit": audit_dir / "audit.json",
    }
    sources = {p.resolve() for p in (features_path, clinical_path, split_path)}
    if len({p.resolve() for p in destinations.values()}) != len(destinations):
        raise ValueError("Nodal output paths must be distinct.")
    if sources & {p.resolve() for p in destinations.values()}:
        raise ValueError("Nodal outputs must not overwrite a source or split file.")
    for path in (features_path, clinical_path, split_path):
        _validate_csv_headers(path)
    dataset = load_dataset(features_path, feature_mode="original")
    _patient_ids(dataset.frame, "patient_id")
    splits = pd.read_csv(split_path, dtype={"patient_id": str})
    splits["patient_id"] = _patient_ids(splits, "patient_id")
    if ("split" not in splits or set(splits.split) != {"development", "test"}
            or set(splits.patient_id) != set(dataset.groups)):
        raise ValueError("Preserve one development/test assignment for every source patient.")
    clinical = pd.read_csv(clinical_path, dtype={"Case ID": str})
    clinical.columns = clinical.columns.str.strip()
    required = {"Case ID", "Histology", "Pathological N stage", "CT Date",
                "Days between CT and surgery"}
    if not required <= set(clinical):
        raise ValueError(f"Clinical source is missing fields: {sorted(required - set(clinical))}")
    clinical["Case ID"] = _patient_ids(clinical, "Case ID")
    source = dataset.frame[["patient_id", "sample_id", "label"]].merge(
        clinical, left_on="patient_id", right_on="Case ID", how="left",
        validate="one_to_one", indicator=True)
    if not source._merge.eq("both").all():
        raise ValueError("Every radiomics patient needs one matching clinical record.")
    if not all(_label_matches(a, b) for a, b in zip(source.label, source.Histology)):
        raise ValueError("Source histology and radiomics patient labels do not match.")
    stages = source["Pathological N stage"].astype("string").str.strip().str.upper()
    dates = pd.to_datetime(source["CT Date"], format="mixed", errors="coerce")
    interval = pd.to_numeric(source["Days between CT and surgery"], errors="coerce")
    known = stages.isin(["N0", "N1", "N2", "N3"])
    reasons = [[] for _ in range(len(source))]
    for mask, reason in [
        (~known, "missing_or_unknown_pathological_n_stage"),
        (dates.isna(), "missing_or_invalid_ct_date"),
        (~interval.between(1, 90), "ct_outside_provisional_1_to_90_day_window"),
        (interval.notna() & interval.mod(1).ne(0), "noninteger_ct_to_surgery_interval"),
    ]:
        for index in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons[index].append(reason)
    eligible = np.asarray([not row for row in reasons])
    target = pd.Series(pd.NA, index=source.index, dtype="Int64")
    target.loc[known] = stages.loc[known].ne("N0").astype(int)
    outcomes = pd.DataFrame({
        "patient_id": source.patient_id, "sample_id": source.sample_id,
        "pathological_n_stage": stages, "nodal_target": target,
        "source_split": source.patient_id.map(splits.set_index("patient_id").split),
        "ct_to_surgery_days": interval, "eligible": eligible,
        "exclusion_reasons": [";".join(row) for row in reasons],
    })
    feature_names = dataset.X.columns.tolist()
    inputs = pd.concat([dataset.frame[["patient_id", "sample_id"]], dataset.X], axis=1)
    cohort = inputs.merge(outcomes[["patient_id", "nodal_target", "source_split"]],
                          on="patient_id", validate="one_to_one")
    cohort = cohort.loc[eligible].sort_values("patient_id").reset_index(drop=True)
    cohort["nodal_target"] = cohort.nodal_target.astype(int)
    eligible_inputs = cohort[["patient_id", "sample_id", *feature_names]]
    eligible_splits = splits.loc[splits.patient_id.isin(cohort.patient_id)].sort_values("patient_id")
    evidence = (surgery_date_evidence or "").strip()
    counts = {
        "source_patients": len(source), "known_nodal_labels": int(known.sum()),
        "source_node_negative": int(target.eq(0).sum()),
        "source_node_positive": int(target.eq(1).sum()),
        "eligible_patients": len(cohort), "excluded_patients": int((~eligible).sum()),
        "eligible_node_negative": int(cohort.nodal_target.eq(0).sum()),
        "eligible_node_positive": int(cohort.nodal_target.eq(1).sum()),
        "source_n_stage_counts": stages.fillna("Unknown").value_counts().to_dict(),
        "eligible_by_split": {
            split: {"node_negative": int((cohort.source_split.eq(split) & cohort.nodal_target.eq(0)).sum()),
                    "node_positive": int((cohort.source_split.eq(split) & cohort.nodal_target.eq(1)).sum())}
            for split in ("development", "test")},
        "exclusions_by_reason": {reason: sum(reason in row for row in reasons)
                                  for reason in sorted({item for row in reasons for item in row})},
    }
    training_allowed = bool(evidence and len(cohort) and cohort.nodal_target.nunique() == 2)
    provenance = {
        "schema_version": 1, "task": "lymph_node_involvement",
        "endpoint": "pathological N0 versus N1/N2/N3",
        "source_features_path": str(features_path), "source_features_sha256": _sha256(features_path),
        "source_clinical_path": str(clinical_path), "source_clinical_sha256": _sha256(clinical_path),
        "source_split_sha256": _sha256(split_path), "feature_names": feature_names,
        "scan_timing_semantics_verified": bool(evidence), "scan_timing_evidence": evidence or None,
        "ct_surgery_window_days": [1, 90],
        "excluded_predictors": ["histology label", "pathological T/N/M stage", "nodal_target",
                                "treatment", "recurrence", "survival", "molecular fields", "patient IDs"],
        "limitations": ["Small retrospective cohort; needs independent validation.",
                        "Clinical N stage is unavailable; occult spread cannot be established.",
                        "Patient labels do not identify affected nodes or distant-organ spread.",
                        "Third-party radiomic extraction requires matched preprocessing."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("cohort", cohort), ("features", eligible_inputs),
                        ("outcomes", outcomes), ("splits", eligible_splits), ("case_audit", outcomes)]:
        frame.to_csv(destinations[name], index=False)
    for name in ("cohort", "features", "outcomes", "splits"):
        provenance[f"{name}_sha256"] = _sha256(destinations[name])
    destinations["provenance"].write_text(json.dumps(provenance, indent=2))
    audit = {
        "schema_version": 1, "task": "lymph_node_involvement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_allowed": training_allowed,
        "status": "ready_for_training" if training_allowed else "blocked_pending_scan_timing_or_label_support",
        "blocking_reason": ({"scan_timing": TIMING_NOTE} if not evidence else {}),
        "semantic_evidence": {"scan_timing": evidence or None}, "counts": counts,
        "cohort_sha256": provenance["cohort_sha256"],
        "outputs": {name: str(path) for name, path in destinations.items()},
        "limitations": provenance["limitations"],
    }
    if cohort.nodal_target.nunique() < 2:
        audit["blocking_reason"]["labels"] = "Both node-negative and node-positive patients are required."
    destinations["audit"].write_text(json.dumps(audit, indent=2))
    return audit


def load_nodal_cohort(path=DEFAULT_COHORT):
    path = Path(path)
    provenance_path = path.with_suffix(".provenance.json")
    if not provenance_path.is_file():
        raise ValueError("Nodal provenance is required before fitting.")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("task") != "lymph_node_involvement" or provenance.get("schema_version") != 1:
        raise ValueError("Unsupported nodal cohort provenance.")
    if _sha256(path) != provenance.get("cohort_sha256"):
        raise ValueError("Nodal cohort changed after preparation.")
    if not provenance.get("scan_timing_semantics_verified") or not provenance.get("scan_timing_evidence"):
        raise ValueError("Nodal training is blocked: document the radiomic CT match and preoperative scan timing.")
    _validate_csv_headers(path)
    rows = pd.read_csv(path, dtype={"patient_id": str, "sample_id": str})
    required = {"patient_id", "sample_id", "nodal_target", "source_split"}
    if not required <= set(rows):
        raise ValueError("Nodal cohort is missing identifiers, targets or split assignments.")
    _patient_ids(rows, "patient_id")
    _patient_ids(rows, "sample_id")
    if not rows.nodal_target.isin([0, 1]).all() or rows.nodal_target.nunique() != 2:
        raise ValueError("Nodal targets must contain both binary classes.")
    if set(rows.source_split) != {"development", "test"}:
        raise ValueError("Preserved development and test patients are required.")
    split_path = path.with_suffix(".splits.csv")
    if _sha256(split_path) != provenance.get("splits_sha256"):
        raise ValueError("Nodal split changed after preparation.")
    splits = pd.read_csv(split_path, dtype={"patient_id": str})
    splits["patient_id"] = _patient_ids(splits, "patient_id")
    if (set(splits.patient_id) != set(rows.patient_id)
            or not rows.patient_id.map(splits.set_index("patient_id").split).eq(rows.source_split).all()):
        raise ValueError("Nodal cohort and preserved split disagree.")
    names = provenance.get("feature_names", [])
    if not names or any(not name.startswith("original_") for name in names) or len(names) != len(set(names)):
        raise ValueError("Nodal predictors must be unique original radiomic feature names.")
    numeric_features(rows, names)
    return rows, provenance
