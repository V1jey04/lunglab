"""Freeze and evaluate competing-risk recurrence models on the saved split."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from .recurrence_data import (DEFAULT_AUDIT, PRIMARY_HORIZON_DAYS,
                              SECONDARY_HORIZON_DAYS, load_recurrence_cohort)
from .recurrence_evaluation import (calibration_by_tertile, horizon_metrics,
                                    reference_horizon_metrics)
from .recurrence_models import (AalenJohansenReference, CensoringDistribution,
                                RecurrenceCoxPair, _require_event_support,
                                ipcw_brier_score)

CONFIG_PATH = Path(__file__).with_name("configs") / "recurrence.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SEED = int(CONFIG["seed"])
OUTER_FOLDS = int(CONFIG["outer_folds"])
INNER_FOLDS = int(CONFIG["inner_folds"])
MINIMUM_EVENTS = int(CONFIG["minimum_events_per_cause_in_training_fold"])
MINIMUM_CENSORING_SURVIVAL = float(CONFIG["minimum_censoring_survival"])
MINIMUM_AT_RISK = int(CONFIG["minimum_patients_at_risk"])
ALPHAS = tuple(float(value) for value in CONFIG["alpha_grid"])
PCA_COMPONENTS = tuple(int(value) for value in CONFIG["pca_component_grid"])
HORIZONS = tuple(int(value) for value in CONFIG["horizons_days"])
if (int(CONFIG["schema_version"]) != 1
        or int(CONFIG["primary_horizon_days"]) != PRIMARY_HORIZON_DAYS
        or int(CONFIG["secondary_horizon_days"]) != SECONDARY_HORIZON_DAYS
        or HORIZONS != (SECONDARY_HORIZON_DAYS, PRIMARY_HORIZON_DAYS)):
    raise ValueError("lungml/configs/recurrence.yaml does not match the supported recurrence protocol.")


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_frozen_models(output: Path, selection: dict) -> None:
    """Check the complete artifact set before loading any model or scoring."""
    frozen = selection.get("frozen_models", {})
    for name in ("clinical_cox", "radiomics_cox", "aalen_johansen_reference"):
        expected = frozen.get(name, {}).get("model_sha256")
        if not expected:
            raise ValueError(f"Frozen recurrence selection is missing a model hash for {name}.")
        path = output / f"{name}.joblib"
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"Frozen recurrence model changed or is missing: {name}.")


def _training_censoring(rows: pd.DataFrame) -> CensoringDistribution:
    return CensoringDistribution(
        min_survival=MINIMUM_CENSORING_SURVIVAL,
        min_at_risk=MINIMUM_AT_RISK,
    ).fit(rows.time_days, rows.event_type)


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("lung-radiomics", "numpy", "pandas", "scikit-learn", "scipy", "joblib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _event_folds(event_type, n_splits: int, seed: int):
    event_type = np.asarray(event_type, dtype=np.int8)
    counts = np.bincount(event_type, minlength=3)
    if counts.min() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} patients of each first-event type per fold; "
            f"found censor/recurrence/death={counts.tolist()}."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(event_type)), event_type))
    for train, _ in folds:
        _require_event_support(event_type[train], minimum=MINIMUM_EVENTS)
    return folds


def _candidate_parameters(family: str):
    if family == "clinical_cox":
        return [{"alpha": alpha, "n_components": 0} for alpha in ALPHAS]
    if family == "radiomics_cox":
        return [{"alpha": alpha, "n_components": components}
                for components in PCA_COMPONENTS for alpha in ALPHAS]
    raise ValueError(f"Unknown recurrence model family: {family}")


def _tie_key(parameters):
    # When validation scores tie, use fewer components and stronger shrinkage.
    return (parameters["n_components"], -parameters["alpha"])


def _fit_pair(rows: pd.DataFrame, parameters: dict) -> RecurrenceCoxPair:
    model = RecurrenceCoxPair(**parameters)
    model.fit(rows, rows.time_days.to_numpy(), rows.event_type.to_numpy())
    return model


def _predict_pair(model: RecurrenceCoxPair, rows: pd.DataFrame) -> dict:
    return model.predict_curves(rows, HORIZONS)


def _supported_horizons(rows: pd.DataFrame) -> list[int]:
    censoring = _training_censoring(rows)
    return [horizon for horizon in HORIZONS
            if censoring.survival_probability(horizon) >= censoring.min_survival
            and np.count_nonzero(censoring.time_ >= horizon) >= censoring.min_at_risk]


def _paired_bootstrap_brier(time, event_type, risks: dict[str, np.ndarray],
                            horizon: int, censoring: CensoringDistribution,
                            *, seed: int = SEED, repetitions: int = 1000) -> dict:
    """Conditional patient bootstrap for paired model Brier-score comparisons."""
    time = np.asarray(time, dtype=float)
    event_type = np.asarray(event_type, dtype=np.int8)
    risks = {name: np.asarray(values, dtype=float) for name, values in risks.items()}
    rng = np.random.default_rng(seed)
    scores = {name: [] for name in risks}
    differences = []
    model_names = list(risks)
    if len(model_names) < 2 or not len(time):
        return {"available": False, "reason": "At least two models and one test patient are required."}
    for _ in range(repetitions):
        sample = rng.integers(0, len(time), size=len(time))
        try:
            replicate = {
                name: ipcw_brier_score(time[sample], event_type[sample], values[sample],
                                       horizon, censoring)
                for name, values in risks.items()
            }
        except ValueError:
            continue
        for name, score in replicate.items():
            scores[name].append(score)
        differences.append(replicate[model_names[1]] - replicate[model_names[0]])
    if not differences:
        return {"available": False, "reason": "No bootstrap replicate had supported censoring weights."}
    model_intervals = {
        name: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for name, values in scores.items() if values
    }
    return {
        "available": True,
        "repetitions_requested": int(repetitions),
        "repetitions_available": len(differences),
        "model_order": model_names,
        "model_brier_ci95": model_intervals,
        "difference_model_2_minus_model_1_ci95": [
            float(value) for value in np.percentile(differences, [2.5, 97.5])],
        "interpretation": "Conditional on the frozen models and development-estimated censoring distribution; does not include model-selection uncertainty.",
    }


def _select_parameters(rows: pd.DataFrame, family: str, seed: int = SEED) -> tuple[dict, list[dict]]:
    event_type = rows.event_type.to_numpy(dtype=np.int8)
    folds = _event_folds(event_type, INNER_FOLDS, seed)
    scores = []
    for parameters in _candidate_parameters(family):
        fold_metrics = []
        for training, validation in folds:
            train_rows = rows.iloc[training]
            valid_rows = rows.iloc[validation]
            model = _fit_pair(train_rows, parameters)
            predictions = _predict_pair(model, valid_rows)
            censoring = _training_censoring(train_rows)
            metrics = horizon_metrics(
                valid_rows.time_days, valid_rows.event_type, predictions,
                PRIMARY_HORIZON_DAYS, censoring, require_supported=True)
            fold_metrics.append(metrics)
        score = float(np.mean([metric["ipcw_brier_score"] for metric in fold_metrics]))
        scores.append({"parameters": parameters, "inner_ipcw_brier_730": score,
                       "folds": fold_metrics})
    selected = min(scores, key=lambda result: (
        result["inner_ipcw_brier_730"], *_tie_key(result["parameters"])))
    return selected["parameters"], scores


def _cross_validated_comparisons(rows: pd.DataFrame, seed: int = SEED) -> dict:
    event_type = rows.event_type.to_numpy(dtype=np.int8)
    outer = _event_folds(event_type, OUTER_FOLDS, seed)
    predictions = {family: np.full((len(rows), len(HORIZONS)), np.nan)
                   for family in ("aalen_johansen", "clinical_cox", "radiomics_cox")}
    fold_scores = {family: [] for family in predictions}
    fold_selections = {family: [] for family in ("clinical_cox", "radiomics_cox")}

    for fold_number, (training, validation) in enumerate(outer, start=1):
        train_rows, valid_rows = rows.iloc[training], rows.iloc[validation]
        reference_scores = []
        censoring = _training_censoring(train_rows)
        for horizon_index, horizon in enumerate(HORIZONS):
            reference = reference_horizon_metrics(
                train_rows.time_days.to_numpy(), train_rows.event_type.to_numpy(),
                valid_rows.time_days.to_numpy(), valid_rows.event_type.to_numpy(),
                horizon, censoring)
            predictions["aalen_johansen"][validation, horizon_index] = reference["prediction"]["recurrence_cif"]
            reference_scores.append(reference)
        fold_scores["aalen_johansen"].append(reference_scores)

        for family in ("clinical_cox", "radiomics_cox"):
            parameters, search = _select_parameters(train_rows, family, seed + fold_number)
            model = _fit_pair(train_rows, parameters)
            curves = _predict_pair(model, valid_rows)
            metrics = [horizon_metrics(
                valid_rows.time_days, valid_rows.event_type, curves, horizon, censoring)
                for horizon in HORIZONS]
            for horizon_index, horizon in enumerate(HORIZONS):
                predictions[family][validation, horizon_index] = \
                    curves["by_horizon"][horizon]["recurrence_cif"]
            fold_scores[family].append(metrics)
            fold_selections[family].append({"fold": fold_number, "parameters": parameters,
                                             "inner_search": search})

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Nested recurrence CV did not produce one prediction per development patient.")
    summary = {}
    for family, values in predictions.items():
        summary[family] = {
            "nested_ipcw_brier_by_horizon": {},
            "nested_competing_risk_auc_by_horizon": {},
            "fold_metrics": fold_scores[family],
            "unavailable_horizons": {},
        }
        for horizon_index, horizon in enumerate(HORIZONS):
            for metric_key, output_key in (
                ("ipcw_brier_score", "nested_ipcw_brier_by_horizon"),
                ("competing_risk_time_dependent_auc", "nested_competing_risk_auc_by_horizon"),
            ):
                available = [fold[horizon_index][metric_key] for fold in fold_scores[family]
                             if fold[horizon_index][metric_key] is not None]
                fully_supported = len(available) == len(fold_scores[family])
                summary[family][output_key][str(horizon)] = (
                    float(np.mean(available)) if fully_supported else None)
                if not fully_supported:
                    summary[family]["unavailable_horizons"][str(horizon)] = (
                        "At least one outer fold lacked the required horizon metric support."
                    )
        for horizon_index, horizon in enumerate(HORIZONS):
            if summary[family]["nested_ipcw_brier_by_horizon"][str(horizon)] is None:
                summary[family][f"calibration_tertiles_{horizon}"] = []
                values[:, horizon_index] = np.nan
            else:
                summary[family][f"calibration_tertiles_{horizon}"] = calibration_by_tertile(
                    rows.time_days.to_numpy(), event_type, values[:, horizon_index], horizon)
    paired_differences = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        differences = []
        for fold_index in range(OUTER_FOLDS):
            clinical_score = fold_scores["clinical_cox"][fold_index][horizon_index]["ipcw_brier_score"]
            radiomics_score = fold_scores["radiomics_cox"][fold_index][horizon_index]["ipcw_brier_score"]
            if clinical_score is not None and radiomics_score is not None:
                differences.append(float(radiomics_score - clinical_score))
        paired_differences[str(horizon)] = {
            "estimand": "radiomics Cox minus clinical Cox; negative favors radiomics",
            "per_outer_fold": differences,
            "mean": float(np.mean(differences)) if differences else None,
            "available_folds": len(differences),
        }
    summary["paired_brier_difference_radiomics_minus_clinical"] = paired_differences
    summary["fold_selections"] = fold_selections
    summary["development_predictions"] = predictions
    return summary


def run_recurrence(cohort_path="data/recurrence_cohort.csv",
                   output="artifacts/recurrence", evaluate=False,
                   audit_path=DEFAULT_AUDIT / "audit.json"):
    cohort_path, output, audit_path = Path(cohort_path), Path(output), Path(audit_path)
    if (output / "selection.json").is_file():
        if evaluate:
            return evaluate_recurrence(cohort_path, output)
        raise ValueError("Frozen recurrence selection already exists; retain the original experiment.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Recurrence output directory is not empty: {output}")
    if not audit_path.is_file():
        raise ValueError(f"A recurrence audit is required before fitting: {audit_path}")
    audit = json.loads(audit_path.read_text())
    audited_cohort = Path(audit.get("outputs", {}).get("cohort", "")).resolve()
    if audited_cohort != cohort_path.resolve():
        raise ValueError("The recurrence audit was generated for a different cohort path.")
    if not audit.get("training_allowed", False):
        raise ValueError(
            "Recurrence training is blocked by the cohort audit. Verify the index-date/scan match "
            "and recurrence-free follow-up definition, then regenerate the cohort provenance."
        )
    rows, provenance = load_recurrence_cohort(cohort_path, require_verified_followup=True)
    split_path = cohort_path.with_suffix(".splits.csv")
    split = pd.read_csv(split_path, dtype={"patient_id": str})
    if (set(split.patient_id) != set(rows.patient_id)
            or split.patient_id.duplicated().any()
            or not split.patient_id.isin(rows.patient_id).all()
            or not set(split.split) <= {"development", "test"}):
        raise ValueError("Recurrence split does not match the prepared cohort.")
    split_by_patient = split.set_index("patient_id").split
    if not rows.patient_id.map(split_by_patient).eq(rows.source_split).all():
        raise ValueError("Recurrence cohort split labels differ from the preserved split manifest.")
    development = rows.loc[rows.source_split.eq("development")].reset_index(drop=True)
    test = rows.loc[rows.source_split.eq("test")].reset_index(drop=True)
    if development.patient_id.isin(test.patient_id).any() or test.empty:
        raise ValueError("Development and preserved test patients must be distinct and nonempty.")
    _require_event_support(development.event_type.to_numpy(dtype=np.int8))
    supported_horizons = _supported_horizons(development)
    if PRIMARY_HORIZON_DAYS not in supported_horizons:
        raise ValueError("The locked 730-day primary horizon is unsupported in development data.")

    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "task": "first recurrence before death after initial surgery",
        "endpoint": "recurrence cumulative incidence with death before recurrence as a competing event",
        "primary_horizon_days": PRIMARY_HORIZON_DAYS,
        "secondary_horizon_days": SECONDARY_HORIZON_DAYS,
        "supported_horizons_days": supported_horizons,
        "ct_surgery_window_days": list(CONFIG["ct_surgery_window_days"]),
        "followup_semantics_verified": True,
        "surgery_date_semantics_verified": True,
        "seed": SEED, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "minimum_events_per_cause_in_each_training_fold": MINIMUM_EVENTS,
        "minimum_training_censoring_survival_at_horizon": MINIMUM_CENSORING_SURVIVAL,
        "minimum_training_patients_at_risk_at_horizon": MINIMUM_AT_RISK,
        "protocol_config_sha256": _hash(CONFIG_PATH),
        "alpha_grid": list(ALPHAS), "pca_component_grid": list(PCA_COMPONENTS),
        "selection_metric": str(CONFIG["selection_metric"]),
        "tie_breaking": str(CONFIG["tie_breaking"]),
        "paired_model_selection": "Select radiomics only if nested development IPCW Brier score beats clinical Cox; otherwise clinical Cox.",
        "evaluation": "Preserved test patients were previously inspected in subtype experiments; recurrence scores are exploratory.",
        "cohort_sha256": _hash(cohort_path),
        "split_sha256": _hash(split_path),
        "outcome_provenance_sha256": _hash(cohort_path.with_suffix(".provenance.json")),
        "source_features_sha256": provenance["source_features_sha256"],
        "source_clinical_sha256": provenance["source_clinical_sha256"],
        "code_sha256": {
            name: _hash(Path(__file__).with_name(name))
            for name in ("recurrence_data.py", "recurrence_models.py",
                         "recurrence_evaluation.py", "recurrence_train.py",
                         "recurrence_predict.py", "cli.py")
        },
        "dependency_versions": _dependency_versions(),
        "development_patients": int(len(development)), "test_patients": int(len(test)),
        "development_events": {
            "recurrence": int(development.event_type.eq(1).sum()),
            "competing_death": int(development.event_type.eq(2).sum()),
            "censored": int(development.event_type.eq(0).sum()),
        },
        "test_events": {
            "recurrence": int(test.event_type.eq(1).sum()),
            "competing_death": int(test.event_type.eq(2).sum()),
            "censored": int(test.event_type.eq(0).sum()),
        },
        "preselected_features": ["clinical_age_at_diagnosis", "log1p(tumor_volume_source_units)"],
        "radiomics_feature_family": "original non-shape image features, correlation pruned then PCA; shape volume excluded from PCA",
        "models": {}, "test_evaluated": False,
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2))

    comparisons = _cross_validated_comparisons(development)
    selection = {**protocol,
                 "development_metrics": {
                     name: {key: value for key, value in metrics.items()
                            if key not in {"fold_metrics", "calibration_tertiles_365",
                                           "calibration_tertiles_730"}}
                     for name, metrics in comparisons.items()
                     if name in {"aalen_johansen", "clinical_cox", "radiomics_cox"}
                 },
                 "development_calibration": {
                     name: {"365": metrics.get("calibration_tertiles_365", []),
                            "730": metrics.get("calibration_tertiles_730", [])}
                     for name, metrics in comparisons.items()
                     if name in {"aalen_johansen", "clinical_cox", "radiomics_cox"}
                 },
                 "fold_selections": comparisons["fold_selections"]}
    clinical_score = selection["development_metrics"]["clinical_cox"][
        "nested_ipcw_brier_by_horizon"][str(PRIMARY_HORIZON_DAYS)]
    radiomics_score = selection["development_metrics"]["radiomics_cox"][
        "nested_ipcw_brier_by_horizon"][str(PRIMARY_HORIZON_DAYS)]
    if clinical_score is None or radiomics_score is None:
        raise ValueError("The locked 730-day primary metric is unsupported in nested development CV.")
    winner = "radiomics_cox" if radiomics_score < clinical_score else "clinical_cox"
    selection["selected_model_by_development"] = winner
    selection["radiomics_incremental_value_demonstrated_in_development"] = bool(
        radiomics_score < clinical_score)
    selection["test_patient_ids_excluded_from_selection"] = test.patient_id.tolist()
    selection_path = output / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2))

    final_models = {}
    for family in ("clinical_cox", "radiomics_cox"):
        parameters, search = _select_parameters(development, family)
        model = _fit_pair(development, parameters)
        bundle = {
            "task": "recurrence_competing_risk",
            "schema_version": 1,
            "model_family": family,
            "model": model,
            "feature_names": list(model.feature_names_in_),
            "feature_schema": {
                "clinical_age_at_diagnosis": "age in years at histological diagnosis; missing values are median-imputed",
                "tumor_volume_source_units": "published original_shape_MeshVolume value; source unit interpretation is not independently verified",
                "original_*": "non-shape radiomic feature columns from the matching source extraction",
                "required_columns": list(model.feature_names_in_),
            },
            "alpha": parameters["alpha"],
            "n_components": parameters["n_components"],
            "horizons_days": supported_horizons,
            "index_definition": "days after provisional initial-surgery date",
            "primary_endpoint": "first recorded recurrence",
            "competing_event": "death before recurrence",
            "training_patient_ids": development.patient_id.tolist(),
            "data_provenance": provenance,
            "prediction_context": "research cohort; not a treatment recommendation",
        }
        model_path = output / f"{family}.joblib"
        joblib.dump(bundle, model_path)
        final_models[family] = {"parameters": parameters, "inner_search": search,
                                "model_sha256": _hash(model_path)}
    aj = AalenJohansenReference().fit(development.time_days, development.event_type)
    aj_path = output / "aalen_johansen_reference.joblib"
    joblib.dump(aj, aj_path)
    selection["frozen_models"] = {**final_models, "aalen_johansen_reference": {"model_sha256": _hash(aj_path)}}
    selection_path.write_text(json.dumps(selection, indent=2))
    (output / "development_metrics.json").write_text(json.dumps(selection["development_metrics"], indent=2))
    pd.DataFrame({"patient_id": development.patient_id,
                  "time_days": development.time_days,
                  "event_type": development.event_type,
                  **{f"{name}_risk_{horizon}d": predictions[:, i]
                     for name, predictions in comparisons["development_predictions"].items()
                     for i, horizon in enumerate(HORIZONS)}}).to_csv(
                         output / "development_oof_predictions.csv", index=False)
    if evaluate:
        return evaluate_recurrence(cohort_path, output)
    return selection


def predict_bundle(bundle: dict, rows: pd.DataFrame) -> dict:
    if bundle.get("task") != "recurrence_competing_risk" or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported recurrence model bundle.")
    return bundle["model"].predict_curves(rows, bundle["horizons_days"])


def evaluate_recurrence(cohort_path: str | Path, output: str | Path) -> dict:
    cohort_path, output = Path(cohort_path), Path(output)
    report_path = output / "external_report.json"
    if report_path.is_file():
        raise ValueError("The recurrence test cohort was already scored; retain the original report.")
    selection_path = output / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Freeze recurrence model selection before evaluating the test cohort.")
    selection = json.loads(selection_path.read_text())
    if _hash(cohort_path) != selection["cohort_sha256"]:
        raise ValueError("Recurrence cohort changed after model selection.")
    split_path = cohort_path.with_suffix(".splits.csv")
    if _hash(split_path) != selection["split_sha256"]:
        raise ValueError("Preserved recurrence split changed after model selection.")
    if _hash(cohort_path.with_suffix(".provenance.json")) != selection["outcome_provenance_sha256"]:
        raise ValueError("Recurrence outcome provenance changed after model selection.")
    _verify_frozen_models(output, selection)
    rows, _ = load_recurrence_cohort(cohort_path, require_verified_followup=True)
    test = rows.loc[rows.source_split.eq("test")].reset_index(drop=True)
    if set(test.patient_id) != set(selection["test_patient_ids_excluded_from_selection"]):
        raise ValueError("The preserved recurrence test patients changed after selection.")
    development = rows.loc[rows.source_split.eq("development")].reset_index(drop=True)
    # Evaluation follows the support policy frozen with this experiment,
    # even if defaults have since changed in the installed configuration.
    censoring = CensoringDistribution(
        min_survival=selection["minimum_training_censoring_survival_at_horizon"],
        min_at_risk=selection["minimum_training_patients_at_risk_at_horizon"],
    ).fit(development.time_days, development.event_type)
    report = {**selection, "test_evaluated": True,
              "evaluation": selection["evaluation"], "test_metrics": {}, "test_predictions": {}}
    prediction_frames = []
    test_risks = {family: {} for family in ("clinical_cox", "radiomics_cox")}
    for family in ("clinical_cox", "radiomics_cox"):
        bundle = joblib.load(output / f"{family}.joblib")
        training_ids = set(bundle["training_patient_ids"])
        if training_ids & set(test.patient_id):
            raise ValueError(f"Test-patient leakage found in {family}.")
        if training_ids != set(development.patient_id):
            raise ValueError(f"Training patients in {family} differ from the frozen development cohort.")
        curves = predict_bundle(bundle, test)
        supported = set(bundle["horizons_days"])
        family_metrics = {}
        for horizon in HORIZONS:
            if horizon not in supported:
                family_metrics[str(horizon)] = {
                    "horizon_days": horizon,
                    "available": False,
                    "unavailable_reason": "Horizon failed the development censoring-support rule.",
                    "ipcw_brier_score": None,
                    "competing_risk_time_dependent_auc": None,
                }
                continue
            test_risks[family][horizon] = np.asarray(
                curves["by_horizon"][horizon]["recurrence_cif"], dtype=float)
            family_metrics[str(horizon)] = horizon_metrics(
                test.time_days.to_numpy(), test.event_type.to_numpy(), curves,
                horizon, censoring)
        report["test_metrics"][family] = family_metrics
        frame = pd.DataFrame({"patient_id": test.patient_id,
                              "time_days": test.time_days,
                              "event_type": test.event_type})
        for horizon in bundle["horizons_days"]:
            values = curves["by_horizon"][horizon]
            frame[f"{family}_recurrence_cif_{horizon}d"] = values["recurrence_cif"]
            frame[f"{family}_death_cif_{horizon}d"] = values["death_cif"]
            frame[f"{family}_event_free_{horizon}d"] = values["event_free_survival"]
        prediction_frames.append(frame)
    aj = joblib.load(output / "aalen_johansen_reference.joblib")
    aj_metrics = {}
    aj_frame = pd.DataFrame({"patient_id": test.patient_id,
                             "time_days": test.time_days,
                             "event_type": test.event_type})
    supported_horizons = set(selection["supported_horizons_days"])
    for horizon in HORIZONS:
        if horizon not in supported_horizons:
            aj_metrics[str(horizon)] = {
                "available": False,
                "unavailable_reason": "Horizon failed the development censoring-support rule.",
                "ipcw_brier_score": None,
                "competing_risk_time_dependent_auc": None,
            }
            continue
        values = aj.predict_at(horizon)
        aj_metrics[str(horizon)] = reference_horizon_metrics(
            development.time_days.to_numpy(), development.event_type.to_numpy(),
            test.time_days.to_numpy(), test.event_type.to_numpy(), horizon, censoring)
        for key in ("recurrence_cif", "death_cif", "event_free_survival"):
            aj_frame[f"aalen_johansen_{key}_{horizon}d"] = values[key]
    report["test_metrics"]["aalen_johansen"] = aj_metrics
    prediction_frames.append(aj_frame)
    report["paired_bootstrap_test_brier"] = {
        str(horizon): (_paired_bootstrap_brier(
            test.time_days.to_numpy(), test.event_type.to_numpy(),
            {family: values[horizon] for family, values in test_risks.items()},
            horizon, censoring, seed=int(selection["seed"]) + horizon)
            if horizon in supported_horizons else {
                "available": False,
                "reason": "Horizon failed the development censoring-support rule.",
            })
        for horizon in HORIZONS
    }
    predictions = prediction_frames[0]
    for next_frame in prediction_frames[1:]:
        predictions = predictions.merge(next_frame, on=["patient_id", "time_days", "event_type"],
                                        validate="one_to_one")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    report["test_predictions_file"] = str(output / "test_predictions.csv")
    report["interpretation"] = (
        "The legacy test cohort was inspected during earlier subtype work. Recurrence prediction "
        "metrics are exploratory and do not establish independent generalization or a treatment benefit."
    )
    report_path.write_text(json.dumps(report, indent=2))
    return report
