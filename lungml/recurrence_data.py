"""Prepare a patient-level first-event recurrence cohort and audit its limits."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TASK_LABELS, load_dataset

DEFAULT_FEATURES = Path("data/radiogenomics_subtype.csv")
DEFAULT_CLINICAL = Path("data/raw/nsclc_radiogenomics/clinical.csv")
DEFAULT_AUDIT = Path("artifacts/recurrence_audit")
SURGERY_WINDOW_DAYS = 90
PRIMARY_HORIZON_DAYS = 730
SECONDARY_HORIZON_DAYS = 365

# The public data descriptor documents recurrence status, recurrence date, and
# last-known-alive, but not that the latter is the last recurrence assessment.
# Keep this explicit so no training path can mistake survival follow-up for
# verified recurrence-free follow-up.
RECURRENCE_ASCERTAINMENT_NOTE = (
    "The source has a binary recurrence field and a last-known-alive date, but "
    "does not establish that recurrence was assessed through that date."
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], format="mixed", errors="coerce").dt.normalize()


def _validate_csv_headers(path: str | Path) -> None:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        try:
            headers = [value.strip() for value in next(csv.reader(stream))]
        except StopIteration as error:
            raise ValueError(f"CSV is empty: {path}") from error
    duplicates = sorted({name for name in headers if headers.count(name) > 1})
    if not headers or any(not name for name in headers) or duplicates:
        raise ValueError(f"CSV has empty or duplicate headers: {path}; duplicates={duplicates[:10]}")


def _label_matches(source_label: str, clinical_label: str) -> bool:
    mapping = {
        "Adenocarcinoma": TASK_LABELS["subtype"][0],
        "Squamous cell carcinoma": TASK_LABELS["subtype"][1],
    }
    return mapping.get(clinical_label) == source_label


def prepare_recurrence(features_path: str | Path = DEFAULT_FEATURES,
                       clinical_path: str | Path = DEFAULT_CLINICAL,
                       output: str | Path = "data/recurrence_cohort.csv",
                       audit_dir: str | Path = DEFAULT_AUDIT,
                       surgery_date_evidence: str | None = None,
                       recurrence_followup_evidence: str | None = None) -> dict:
    """Join source records, preserve the legacy split, and write an audit.

    Surgery is provisionally reconstructed as CT date plus the published
    positive ``Days between CT and surgery`` interval. Per-patient date order
    is auditable because the source release retains chronological order after
    date shifting. Contradictory cases are excluded. Recurrence-negative
    follow-up remains explicitly unverified, so the resulting cohort cannot
    yet be used to train a model.
    """
    features_path, clinical_path = Path(features_path), Path(clinical_path)
    output, audit_dir = Path(output), Path(audit_dir)
    surgery_evidence = (surgery_date_evidence or "").strip()
    followup_evidence = (recurrence_followup_evidence or "").strip()
    surgery_semantics_verified = bool(surgery_evidence)
    recurrence_semantics_verified = bool(followup_evidence)
    protected_sources = {features_path.resolve(), clinical_path.resolve(),
                         features_path.with_suffix(".splits.csv").resolve()}
    if output.resolve() in protected_sources:
        raise ValueError("Recurrence cohort output must not overwrite a source or split file.")
    _validate_csv_headers(features_path)
    _validate_csv_headers(clinical_path)
    dataset = load_dataset(features_path, feature_mode="all")
    if dataset.frame.patient_id.duplicated().any():
        raise ValueError("Recurrence cohort requires one radiomic row per patient.")
    split_path = features_path.with_suffix(".splits.csv")
    if not split_path.is_file():
        raise FileNotFoundError(f"Preserved patient split is required: {split_path}")
    splits = pd.read_csv(split_path, dtype={"patient_id": str})
    required_split = {"patient_id", "split"}
    if (not required_split <= set(splits.columns) or splits.patient_id.duplicated().any()
            or set(splits.patient_id) != set(dataset.groups)
            or set(splits.split) != {"development", "test"}):
        raise ValueError("The existing split must assign each source patient exactly once.")

    clinical = pd.read_csv(clinical_path, dtype={"Case ID": str})
    clinical.columns = clinical.columns.str.strip()
    required = {
        "Case ID", "Histology", "Pathological M stage", "Recurrence",
        "Date of Recurrence", "Date of Death", "Date of Last Known Alive",
        "Survival Status", "CT Date", "Days between CT and surgery",
        "Age at Histological Diagnosis",
    }
    if not required <= set(clinical.columns):
        raise ValueError(f"Clinical source is missing columns: {sorted(required - set(clinical.columns))}")
    if clinical["Case ID"].isna().any() or clinical["Case ID"].duplicated().any():
        raise ValueError("Clinical source must have unique, nonmissing patient IDs.")

    radiomics = dataset.frame[["patient_id", "sample_id", "label"]].copy()
    radiomics_features = dataset.X.loc[:, [
        column for column in dataset.X
        if column.startswith("original_") and not column.startswith("original_shape_")
    ]].copy()
    if "original_shape_MeshVolume" not in dataset.frame:
        raise ValueError("The source data do not contain original_shape_MeshVolume.")
    radiomics["tumor_volume_source_units"] = pd.to_numeric(
        dataset.frame["original_shape_MeshVolume"], errors="raise").to_numpy()
    radiomics = pd.concat([radiomics, radiomics_features.reset_index(drop=True)], axis=1)

    clinical = clinical.loc[clinical["Case ID"].isin(radiomics.patient_id)].copy()
    cohort = radiomics.merge(clinical, left_on="patient_id", right_on="Case ID",
                             validate="one_to_one", how="left", indicator=True)
    if len(cohort) != len(radiomics) or not cohort._merge.eq("both").all():
        raise ValueError("Every radiomics patient must have one official clinical record.")
    if not all(_label_matches(a, b) for a, b in zip(cohort.label, cohort.Histology)):
        raise ValueError("Radiomics labels do not match the official clinical histology.")

    for column in ("CT Date", "Date of Recurrence", "Date of Death", "Date of Last Known Alive"):
        cohort[f"_parsed_{column}"] = _date_column(cohort, column)
    interval = pd.to_numeric(cohort["Days between CT and surgery"], errors="coerce")
    ct_date = cohort["_parsed_CT Date"]
    surgery_date = ct_date + pd.to_timedelta(interval, unit="D")
    recurrence_date = cohort["_parsed_Date of Recurrence"]
    death_date = cohort["_parsed_Date of Death"]
    last_alive = cohort["_parsed_Date of Last Known Alive"]
    # Unknown statuses are excluded below; keep their comparisons two-valued
    # so nullable strings cannot interrupt the audit or become event labels.
    rec_label = cohort.Recurrence.astype("string").str.strip().str.casefold().fillna("")
    survival_label = cohort["Survival Status"].astype("string").str.strip().str.casefold().fillna("")
    m_stage = cohort["Pathological M stage"].astype("string").str.strip().str.upper().fillna("")

    reasons: list[list[str]] = [[] for _ in range(len(cohort))]

    def flag(mask: pd.Series | np.ndarray, reason: str) -> None:
        values = np.asarray(mask, dtype=bool)
        for row_index in np.flatnonzero(values):
            reasons[row_index].append(reason)

    flag(~ct_date.notna(), "missing_or_unparseable_ct_date")
    flag(~interval.notna() | ~interval.between(1, 3650), "invalid_ct_to_surgery_interval")
    flag(~rec_label.isin(["yes", "no"]), "missing_or_unknown_recurrence_status")
    flag(~survival_label.isin(["alive", "dead"]), "missing_or_unknown_survival_status")
    flag(~m_stage.isin(["M0", "M1A", "M1B"]), "missing_or_unknown_pathological_m_stage")
    flag(m_stage.isin(["M1A", "M1B"]), "metastatic_at_baseline")
    flag(interval.gt(SURGERY_WINDOW_DAYS), "ct_outside_primary_90_day_window")
    flag((rec_label == "yes") & ~recurrence_date.notna(), "recurrence_without_valid_date")
    flag((survival_label == "dead") & ~death_date.notna(), "death_without_valid_date")
    flag((survival_label == "alive") & death_date.notna(), "alive_with_death_date")
    flag(~last_alive.notna(), "missing_or_unparseable_last_known_alive")
    flag(recurrence_date.notna() & recurrence_date.lt(surgery_date), "recurrence_before_provisional_surgery")
    flag(death_date.notna() & death_date.lt(surgery_date), "death_before_provisional_surgery")
    flag(last_alive.notna() & last_alive.lt(surgery_date), "last_known_alive_before_provisional_surgery")
    flag((rec_label == "no") & recurrence_date.notna(), "no_recurrence_label_with_recurrence_date")
    flag((rec_label == "yes") & ~recurrence_date.notna(), "recurrence_label_without_recurrence_date")

    status_known = rec_label.isin(["yes", "no"]) & survival_label.isin(["alive", "dead"])
    flag(status_known & survival_label.eq("alive") & (rec_label == "yes") & recurrence_date.gt(last_alive),
         "recurrence_after_last_known_alive")
    flag(death_date.notna() & recurrence_date.notna() & recurrence_date.gt(death_date),
         "recurrence_after_death")
    flag((survival_label == "dead") & death_date.notna() & last_alive.notna()
         & last_alive.gt(death_date), "last_known_alive_after_death")
    flag(status_known & (survival_label == "dead") & death_date.lt(surgery_date),
         "death_before_index")

    followup_end = last_alive.copy()
    followup_end.loc[survival_label.eq("dead")] = death_date.loc[survival_label.eq("dead")]
    negative = rec_label.eq("no")
    flag(negative & followup_end.notna() & followup_end.le(surgery_date), "no_positive_followup_after_index")

    # If recurrence and death share a calendar day, use the documented
    # recurrence as the first event. The source has day precision only.
    first_event_date = pd.Series(pd.NaT, index=cohort.index, dtype="datetime64[ns]")
    event_type = np.zeros(len(cohort), dtype=np.int8)  # 0=censored, 1=recurrence, 2=death
    rec_first = (rec_label.eq("yes") & recurrence_date.notna()
                 & (death_date.isna() | recurrence_date.le(death_date)))
    death_first = (survival_label.eq("dead") & death_date.notna() & ~rec_first)
    event_type[rec_first.to_numpy(dtype=bool)] = 1
    event_type[death_first.to_numpy(dtype=bool)] = 2
    first_event_date.loc[rec_first] = recurrence_date.loc[rec_first]
    first_event_date.loc[death_first] = death_date.loc[death_first]
    censor = event_type == 0
    first_event_date.loc[censor] = last_alive.loc[censor]

    eligible_before_followup_gate = np.array([len(row) == 0 for row in reasons])
    time_days = (first_event_date - surgery_date).dt.days.astype("float64")
    bad_time = eligible_before_followup_gate & (~np.isfinite(time_days) | time_days.le(0))
    flag(bad_time, "invalid_first_event_or_censoring_duration")
    eligible = np.array([len(row) == 0 for row in reasons])
    eligible_180 = np.asarray([
        interval.iloc[index] >= 1 and interval.iloc[index] <= 180
        and all(reason != "ct_outside_primary_90_day_window" for reason in row)
        for index, row in enumerate(reasons)
    ], dtype=bool) & np.isfinite(time_days) & (time_days > 0)

    audit_case = pd.DataFrame({
        "patient_id": cohort.patient_id,
        "source_split": cohort.patient_id.map(splits.set_index("patient_id").split),
        "pathological_m_stage": m_stage,
        "ct_to_surgery_days": interval,
        "provisional_surgery_date": surgery_date.dt.strftime("%Y-%m-%d"),
        "event_type": np.asarray(["censored", "recurrence", "death first"])[event_type],
    })
    # Calendar dates stay in this local audit; modeling outputs use durations.
    audit_case["ct_date"] = ct_date.dt.strftime("%Y-%m-%d")
    audit_case["recurrence_date"] = recurrence_date.dt.strftime("%Y-%m-%d")
    audit_case["death_date"] = death_date.dt.strftime("%Y-%m-%d")
    audit_case["last_known_alive"] = last_alive.dt.strftime("%Y-%m-%d")
    audit_case["time_days"] = time_days
    audit_case["eligible_90_day_primary"] = eligible
    audit_case["eligible_180_day_sensitivity"] = eligible_180
    audit_case["followup_semantics_verified"] = recurrence_semantics_verified
    audit_case["surgery_date_semantics_verified"] = surgery_semantics_verified
    audit_case["exclusion_reasons"] = [";".join(row) for row in reasons]

    outcomes = pd.DataFrame({
        "patient_id": cohort.patient_id,
        "sample_id": cohort.sample_id,
        "ct_date": ct_date.dt.strftime("%Y-%m-%d"),
        "index_date": surgery_date.dt.strftime("%Y-%m-%d"),
        "time_days": time_days,
        "event_type": event_type,
        "source_split": cohort.patient_id.map(splits.set_index("patient_id").split),
        "eligible": eligible,
        "exclusion_reasons": [";".join(row) for row in reasons],
        "outcome_source": str(clinical_path),
        "followup_semantics_verified": recurrence_semantics_verified,
    })
    selected_features = radiomics[["patient_id", "sample_id"]].copy()
    selected_features["clinical_age_at_diagnosis"] = pd.to_numeric(
        cohort["Age at Histological Diagnosis"], errors="coerce").to_numpy()
    selected_features["tumor_volume_source_units"] = radiomics["tumor_volume_source_units"].to_numpy()
    selected_features = pd.concat([selected_features, radiomics_features.reset_index(drop=True)], axis=1)
    model_frame = selected_features.merge(
        outcomes.loc[:, ["patient_id", "time_days", "event_type", "source_split", "eligible"]],
        on="patient_id", validate="one_to_one")
    eligible_frame = model_frame.loc[model_frame.eligible].reset_index(drop=True)
    eligible_splits = splits.loc[splits.patient_id.isin(eligible_frame.patient_id)].copy()
    eligible_splits = eligible_splits.sort_values("patient_id").reset_index(drop=True)
    eligible_frame = eligible_frame.sort_values("patient_id").reset_index(drop=True)

    reasons_flat = [item for row in reasons for item in row]
    eligible_outcomes = outcomes.loc[outcomes.eligible]
    counts = {
        "source_patients": int(len(cohort)),
        "source_recurrence_yes": int(rec_label.eq("yes").sum()),
        "source_recurrence_no": int(rec_label.eq("no").sum()),
        "eligible_by_dates_and_clinical_rules": int(eligible.sum()),
        "eligible_development_patients": int(eligible_outcomes.source_split.eq("development").sum()),
        "eligible_test_patients": int(eligible_outcomes.source_split.eq("test").sum()),
        "eligible_recurrence_events": int((eligible_outcomes.event_type == 1).sum()),
        "eligible_competing_deaths": int((eligible_outcomes.event_type == 2).sum()),
        "eligible_right_censored": int((eligible_outcomes.event_type == 0).sum()),
        "sensitivity_180_day": {
            "eligible_patients": int(eligible_180.sum()),
            "recurrence_events": int(np.count_nonzero(eligible_180 & (event_type == 1))),
            "competing_deaths": int(np.count_nonzero(eligible_180 & (event_type == 2))),
            "right_censored": int(np.count_nonzero(eligible_180 & (event_type == 0))),
        },
        "excluded_patients": int((~eligible).sum()),
        "followup_semantics_verified": recurrence_semantics_verified,
        "surgery_date_semantics_verified": surgery_semantics_verified,
        "training_allowed": bool(recurrence_semantics_verified
                                 and surgery_semantics_verified and eligible.sum() > 0),
        "exclusions_by_reason": {reason: int(sum(reason in row for row in reasons))
                                  for reason in sorted(set(reasons_flat))},
        "event_counts_by_split": {
            split: {"recurrence": int(((eligible_outcomes.source_split == split)
                                        & (eligible_outcomes.event_type == 1)).sum()),
                    "competing_death": int(((eligible_outcomes.source_split == split)
                                             & (eligible_outcomes.event_type == 2)).sum()),
                    "censored": int(((eligible_outcomes.source_split == split)
                                     & (eligible_outcomes.event_type == 0)).sum())}
            for split in ("development", "test")
        },
        "days_followed_at_horizon_by_split": {
            split: {str(horizon): int(((eligible_outcomes.source_split == split)
                                      & (eligible_outcomes.time_days >= horizon)).sum())
                    for horizon in (SECONDARY_HORIZON_DAYS, PRIMARY_HORIZON_DAYS)}
            for split in ("development", "test")
        },
    }
    dataset_provenance_path = features_path.with_suffix(".provenance.json")
    dataset_provenance = (json.loads(dataset_provenance_path.read_text())
                          if dataset_provenance_path.is_file() else None)
    provenance = {
        "schema_version": 1,
        "task": "first recurrence before death after initial surgery",
        "source_features_path": str(features_path),
        "source_features_sha256": _sha256(features_path),
        "source_clinical_path": str(clinical_path),
        "source_clinical_sha256": _sha256(clinical_path),
        "source_split_path": str(split_path),
        "source_split_sha256": _sha256(split_path),
        "output_path": str(output),
        "surgery_date_rule": "CT Date plus positive Days between CT and surgery; requires source sign interpretation",
        "primary_preoperative_ct_window_days": [1, SURGERY_WINDOW_DAYS],
        "sensitivity_preoperative_ct_window_days": [1, 180],
        "primary_horizon_days": PRIMARY_HORIZON_DAYS,
        "secondary_horizon_days": SECONDARY_HORIZON_DAYS,
        "same_day_recurrence_death_rule": "recurrence first when recurrence is documented on the death date",
        "recurrence_negative_censoring_rule": "last-known-alive date; unverified as last recurrence assessment",
        "recurrence_free_followup_semantics_verified": recurrence_semantics_verified,
        "surgery_date_semantics_verified": surgery_semantics_verified,
        "surgery_date_evidence": surgery_evidence or None,
        "recurrence_followup_evidence": followup_evidence or None,
        "followup_semantics_note": followup_evidence or RECURRENCE_ASCERTAINMENT_NOTE,
        "feature_columns": ["clinical_age_at_diagnosis", "tumor_volume_source_units",
                            *radiomics_features.columns.tolist()],
        "excluded_predictors": ["histology label", "pathological T/N/M stage", "Recurrence",
                                "Date of Recurrence", "Survival Status", "Date of Death",
                                "Date of Last Known Alive", "treatment and molecular fields"],
        "source_dataset_provenance": dataset_provenance,
        "counts": counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output.with_name("recurrence_outcomes.csv"), index=False)
    eligible_frame.to_csv(output, index=False)
    eligible_features = selected_features.loc[
        selected_features.patient_id.isin(eligible_frame.patient_id)
    ].sort_values("patient_id").reset_index(drop=True)
    features_path = output.with_name("recurrence_features.csv")
    eligible_features.to_csv(features_path, index=False)
    eligible_splits.to_csv(output.with_suffix(".splits.csv"), index=False)
    provenance["output_sha256"] = _sha256(output)
    provenance["feature_table_path"] = str(features_path)
    provenance["feature_table_sha256"] = _sha256(features_path)
    provenance["outcome_table_sha256"] = _sha256(output.with_name("recurrence_outcomes.csv"))
    provenance["split_output_sha256"] = _sha256(output.with_suffix(".splits.csv"))
    (output.with_suffix(".provenance.json")).write_text(json.dumps(provenance, indent=2))
    audit_case.to_csv(audit_dir / "case_audit.csv", index=False)
    audit_json = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("ready_for_training_review" if counts["training_allowed"]
                   else "blocked_pending_outcome_and_index_date_semantics"),
        "training_allowed": counts["training_allowed"],
        "blocking_reason": {
            **({"surgery_date": "The positive CT-to-surgery interval and matching radiomic CT are not independently verified."}
               if not surgery_semantics_verified else {}),
            **({"recurrence_free_followup": RECURRENCE_ASCERTAINMENT_NOTE}
               if not recurrence_semantics_verified else {}),
        },
        "surgery_date_interpretation": provenance["surgery_date_rule"],
        "date_semantics_verified": surgery_semantics_verified,
        "semantic_evidence": {
            "surgery_date": surgery_evidence or None,
            "recurrence_free_followup": followup_evidence or None,
        },
        "counts": counts,
        "source_hashes": {
            "features": provenance["source_features_sha256"],
            "clinical": provenance["source_clinical_sha256"],
            "split": provenance["source_split_sha256"],
        },
        "outputs": {
            "cohort": str(output),
            "outcomes": str(output.with_name("recurrence_outcomes.csv")),
            "features": str(features_path),
            "splits": str(output.with_suffix(".splits.csv")),
            "case_audit": str(audit_dir / "case_audit.csv"),
        },
    }
    (audit_dir / "audit.json").write_text(json.dumps(audit_json, indent=2))
    return audit_json


def load_recurrence_cohort(path: str | Path, *, require_verified_followup: bool = True):
    """Load the cohort and refuse model fitting on unverified negative follow-up."""
    path = Path(path)
    provenance_path = path.with_suffix(".provenance.json")
    if not provenance_path.is_file():
        raise ValueError("Recurrence provenance is required before fitting.")
    provenance = json.loads(provenance_path.read_text())
    if _sha256(path) != provenance.get("output_sha256"):
        raise ValueError("Recurrence cohort hash does not match its provenance.")
    if require_verified_followup and not (
            provenance.get("recurrence_free_followup_semantics_verified", False)
            and provenance.get("surgery_date_semantics_verified", False)):
        raise ValueError(
            "Recurrence training is blocked: verify the CT-to-surgery date interpretation and radiomic "
            "scan match, and verify that recurrence-negative status is assessed through the censoring "
            "date or supply a documented recurrence-assessment date per patient."
        )
    frame = pd.read_csv(path, dtype={"patient_id": str, "sample_id": str})
    required = {"patient_id", "sample_id", "time_days", "event_type", "source_split", "eligible"}
    if not required <= set(frame.columns):
        raise ValueError(f"Recurrence cohort is missing fields: {sorted(required - set(frame.columns))}")
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError("Recurrence training requires unique, nonmissing patient IDs.")
    if not frame.eligible.eq(True).all() or not np.isfinite(frame.time_days).all() or frame.time_days.le(0).any():
        raise ValueError("Recurrence cohort contains ineligible patients or invalid event times.")
    if not set(frame.event_type).issubset({0, 1, 2}):
        raise ValueError("event_type must encode censoring=0, recurrence=1, or death-before-recurrence=2.")
    if set(frame.source_split) != {"development", "test"}:
        raise ValueError("The preserved patient development and test split is required.")
    return frame, provenance
