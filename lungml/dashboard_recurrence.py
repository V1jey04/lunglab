"""Live, explicitly exploratory recorded-recurrence classification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .case_report import build_case_report
from .dashboard_runtime import input_signature, stream_inference
from .explanations import explain_case
from .quality import assess_inputs, load_profile, sha256
from .recurrence_classifier import DEFAULT_OUTPUT, GRADE_VALUES, MODEL_NAMES, verify_recorded_models
from .recurrence_predict import read_feature_csv
from .service import trusted_bundle

FEATURE_LABELS = {
    "clinical_age_at_diagnosis": "Age at diagnosis", "clinical_male": "Recorded sex · male",
    "clinical_pack_years": "Smoking pack-years", "clinical_smoking_never": "Never smoked",
    "clinical_smoking_former": "Former smoker", "clinical_smoking_current": "Current smoker",
    "clinical_pathological_t": "Pathological T category", "clinical_pathological_n": "Pathological N category",
    "clinical_grade": "Tumour grade", "clinical_lymphovascular_invasion": "Lymphovascular invasion",
    "clinical_pleural_invasion": "Pleural invasion",
}


def manual_case():
    st.caption("Use a postoperative profile with known pathology. N stage is an input here; this model does not predict lymph-node involvement.")
    a, b, c = st.columns(3)
    with a:
        patient_id = st.text_input("Recurrence case ID", "NEW-001", key="recorded_manual_id")
        age = st.number_input("Age at diagnosis (years)", min_value=18, max_value=110, value=67, key="recorded_age")
        sex = st.selectbox("Recorded sex", ["Male", "Female"], key="recorded_sex")
        smoking = st.selectbox("Smoking history", ["Former", "Never", "Current"], key="recorded_smoking")
    with b:
        pack_years = st.number_input("Smoking exposure (pack-years)", min_value=0., max_value=250., value=30., step=1.,
                                    help="Packs smoked per day × years smoked.", key="recorded_pack_years")
        t = st.selectbox("Pathological T category", [0, 1, 2, 3, 4], index=2,
                         format_func=lambda v: "Tis" if v == 0 else f"T{v}", key="recorded_t")
        n = st.selectbox("Pathological N category", [0, 1, 2, 3], format_func=lambda v: f"N{v}", key="recorded_n")
    with c:
        grade = st.selectbox("Histological grade", list(GRADE_VALUES), index=2, key="recorded_grade")
        lvi = st.selectbox("Lymphovascular invasion", ["Absent", "Present", "Unknown"], key="recorded_lvi")
        pleural = st.selectbox("Pleural invasion", ["No", "Yes"], key="recorded_pleural")
    if smoking == "Never" and pack_years:
        st.caption("Smoking exposure is set to 0 pack-years for a never-smoker.")
    return pd.DataFrame([{
        "patient_id": patient_id.strip(), "clinical_age_at_diagnosis": float(age), "clinical_male": float(sex == "Male"),
        "clinical_pack_years": 0. if smoking == "Never" else pack_years,
        "clinical_smoking_never": float(smoking == "Never"), "clinical_smoking_former": float(smoking == "Former"),
        "clinical_smoking_current": float(smoking == "Current"), "clinical_pathological_t": float(t),
        "clinical_pathological_n": float(n), "clinical_grade": GRADE_VALUES[grade],
        "clinical_lymphovascular_invasion": {"Absent": 0., "Present": 1., "Unknown": np.nan}[lvi],
        "clinical_pleural_invasion": float(pleural == "Yes"),
    }])


def render_recorded_recurrence():
    """Return whether a trained record-based experiment is present."""
    from .dashboard import cards, chart, live_output, number, panel, pct
    output = DEFAULT_OUTPUT
    report_path = output / "report.json"
    if not report_path.is_file():
        return False
    panel("Run a recurrence prediction", "TRAINED EXPLORATORY MODEL", "Real model · local inference")
    try:
        selection = verify_recorded_models(output)
        report = json.loads(report_path.read_text())
    except (ValueError, OSError) as error:
        st.error(str(error))
        return True
    st.info("This trained classifier estimates recurrence recorded during the source study’s varying follow-up periods. "
            "It uses surgical pathology information. There is no one-year or two-year horizon, and a ‘no recorded recurrence’ output does not guarantee future freedom from recurrence.")
    cards([("Patients with recorded labels", report["development_patients"] + report["test_patients"], "Explicit source yes/no recurrence field"),
           ("Recorded recurrences", report["recorded_recurrences"], "Observed labels used as the target"),
           ("Development patients", report["development_patients"], "Preprocessing and model choice"),
           ("Test patients", report["test_patients"], "Previously inspected patients · exploratory")])
    names = list(selection["frozen_models"])
    name = st.selectbox("Recurrence classifier", names,
                        index=names.index(selection["selected_model_by_development"]),
                        format_func=lambda value: MODEL_NAMES[value], key="recorded_model")
    st.caption(f"Development-selected model: {MODEL_NAMES[selection['selected_model_by_development']]}. "
               "The subtype model picker in the sidebar controls the subtype tabs.")
    path = output / f"{name}.joblib"
    try:
        bundle = trusted_bundle({"path": path, "task": "recorded_recurrence"})
        profile = load_profile(path, bundle)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return True
    sources = ["Example patients", "Enter patient details", "Upload feature CSV"] if name == "clinical_logistic" else ["Example patients", "Upload feature CSV"]
    source = st.radio("Recurrence input", sources, horizontal=True, key="recorded_source")
    frame, known_labels = None, {}
    try:
        if source == "Example patients":
            cohort = Path(selection["dataset_path"])
            if sha256(cohort) != selection["cohort_sha256"]:
                raise ValueError("Example cohort no longer matches the frozen recurrence experiment.")
            examples = read_feature_csv(cohort)
            examples = examples.loc[examples.patient_id.isin(selection["test_patient_ids"])].reset_index(drop=True)
            patient = st.selectbox("Example recurrence patient", examples.patient_id.tolist(), key="recorded_example_patient")
            batch = st.checkbox("Run five example patients", key="recorded_batch")
            chosen = examples.head(5) if batch else examples.loc[examples.patient_id.eq(patient)]
            known_labels = dict(zip(chosen.patient_id, chosen.label))
            frame = chosen[["patient_id"] + bundle["feature_names"]].copy()
            st.caption("These are saved test patients. Replaying them demonstrates inference and provides no new validation.")
        elif source == "Enter patient details":
            frame = manual_case()
        else:
            template = pd.DataFrame(columns=["patient_id"] + bundle["feature_names"]).to_csv(index=False)
            st.download_button("Download recurrence CSV template", template, f"{name}_template.csv", "text/csv", key="recorded_template")
            uploaded = st.file_uploader("Recorded-recurrence feature CSV", type=["csv"], key="recorded_upload")
            if uploaded is not None:
                frame = read_feature_csv(uploaded)
        if frame is not None and not 1 <= len(frame) <= 500:
            raise ValueError("Queue between 1 and 500 recurrence cases.")
    except (ValueError, OSError, KeyError) as error:
        st.session_state.pop("recorded_recurrence_run", None)
        st.error(str(error))
        frame = None
    signature = input_signature(path, frame) if frame is not None else None
    state = st.session_state.get("recorded_recurrence_run")
    if state and state["signature"] != signature:
        st.session_state.pop("recorded_recurrence_run", None)
        state = None
    if st.button("Predict recurrence", type="primary", disabled=frame is None, key="recorded_predict"):
        st.session_state.pop("recorded_recurrence_run", None)
        state = None
        try:
            progress = st.progress(0., text="Validating recurrence inputs")
            for event in stream_inference(bundle, frame):
                progress.progress(event.completed / event.total, text=f"{event.completed} / {event.total} recurrence cases complete")
                if event.stage == "complete":
                    state = {"signature": signature, "result": event.result, "compute_ms": event.compute_ms}
                    st.session_state["recorded_recurrence_run"] = state
        except (ValueError, OSError) as error:
            st.error(str(error))
    if state:
        results = state["result"]
        st.caption(f"Measured model time: {state['compute_ms']:.1f} ms · {len(results)} completed case(s)")
        patient = st.selectbox("Inspect recurrence prediction", results.patient_id.tolist(), key="recorded_inspect")
        row = results.loc[results.patient_id.eq(patient)].iloc[0]
        case = frame.loc[frame.patient_id.eq(patient)]
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            live_output(row, patient, "Recurrence classifier")
        with right:
            st.metric("Recorded recurrence probability", pct(row.p_recorded_recurrence))
            st.metric("Saved research decision threshold", pct(bundle["threshold"]))
            st.write("The classifier predicts recorded recurrence when its probability reaches the saved threshold. "
                     "This threshold was chosen from development predictions; it is not a treatment or surveillance rule.")
            if patient in known_labels:
                st.write("**Source record:** " + ("Recurrence recorded" if known_labels[patient] else "No recurrence recorded"))
            quality = assess_inputs(bundle, case, profile)[0]
            st.caption(f"Input check: {quality['missing_feature_count']} missing values; "
                       f"{quality['outside_reference_count']} outside the training reference range. "
                       + ("Input review suggested." if quality["input_review_suggested"] else "No input review flag."))
        explanation = explain_case(bundle, case, profile)
        with st.expander("Why did the recurrence model give this score?", expanded=True):
            contributions = pd.DataFrame(explanation["contributions"])
            contributions["Feature"] = contributions.feature.map(lambda value: FEATURE_LABELS.get(value, value))
            chart(contributions, {"mark": {"type": "bar", "cornerRadiusEnd": 3}, "encoding": {
                "y": {"field": "Feature", "type": "nominal", "sort": "-x", "title": None},
                "x": {"field": "contribution", "type": "quantitative", "title": "Contribution to recorded-recurrence log odds"},
                "color": {"condition": {"test": "datum.contribution > 0", "value": "#0b9488"}, "value": "#254967"},
                "tooltip": [{"field": "Feature"}, {"field": "contribution", "format": ".3f"}]}}, 280)
            st.caption("Positive contributions raise the score; negative contributions lower it. These are fitted associations, not causes. "
                       "The intercept plus all contributions reconstructs the model probability.")
        st.dataframe(results, hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.download_button("Download recurrence predictions", results.to_csv(index=False), "recorded_recurrence_predictions.csv", "text/csv", key="recorded_csv_export")
        case_report = build_case_report(bundle, case, results.loc[results.patient_id.eq(patient)], sha256(path), [quality], explanation)
        b.download_button("Download recurrence case report", case_report, f"{patient}_recurrence_report.html", "text/html", key="recorded_report_export")
    training_fit_path = Path("artifacts/recurrence_training_fit/report.json")
    if training_fit_path.is_file():
        fit_report = json.loads(training_fit_path.read_text())
        tree_path = training_fit_path.parent / "decision_tree.joblib"
        if (fit_report.get("task") == "recurrence_training_fit" and tree_path.is_file()
                and sha256(tree_path) == fit_report.get("model_sha256")
                and sha256(output / "selection.json") == fit_report.get("source_selection_sha256")):
            with st.container(border=True, key="panel_recurrence_training_fit"):
                panel("Training fit on a smaller cohort", "TRAINING EXPERIMENT", "Decision tree · measured scores")
                cards([("Training accuracy", pct(fit_report["training_metrics"]["accuracy"]),
                        f"{fit_report['training_patients']} patients used to fit this tree"),
                       ("Separate test accuracy", pct(fit_report["test_metrics"]["accuracy"]),
                        f"{fit_report['test_patients']} patients excluded from fitting"),
                       ("Training patients", fit_report["training_patients"], "Fixed subset of development patients"),
                       ("Test patients", fit_report["test_patients"], "Same preserved test partition")])
                st.write("This unrestricted decision tree fits every label in its small training cohort. "
                         "Training accuracy measures fit on those same patients; test accuracy measures the separate test group.")
                comparison = pd.DataFrame([{"Patient group": "Training", "Accuracy": fit_report["training_metrics"]["accuracy"]},
                                           {"Patient group": "Separate test", "Accuracy": fit_report["test_metrics"]["accuracy"]}])
                chart(comparison, {"mark": {"type": "bar", "cornerRadiusEnd": 5}, "encoding": {
                    "y": {"field": "Patient group", "type": "nominal", "title": None, "sort": ["Training", "Separate test"]},
                    "x": {"field": "Accuracy", "type": "quantitative", "scale": {"domain": [0, 1]}, "axis": {"format": ".0%"}},
                    "color": {"field": "Patient group", "type": "nominal", "legend": None,
                              "scale": {"domain": ["Training", "Separate test"], "range": ["#0b9488", "#254967"]}},
                    "tooltip": [{"field": "Patient group"}, {"field": "Accuracy", "format": ".1%"}]}}, 130)
                gap = fit_report["training_metrics"]["accuracy"] - fit_report["test_metrics"]["accuracy"]
                st.caption(f"Overfitting gap: {gap * 100:.1f} percentage points. "
                           "The 100% training score is not a claim of accuracy on new patients. "
                           "This is a separate experiment from the recurrence classifier used above.")
                st.download_button("Download training-fit experiment", training_fit_path.read_bytes(),
                                   "recurrence_training_fit.json", "application/json", key="recurrence_training_fit_export")
        else:
            st.warning("The training-fit experiment no longer matches its saved model or source run.")
    with st.expander("Measured recurrence performance and limitations", expanded=True):
        metrics = report["test_metrics"][name]
        cards([("Test accuracy", pct(metrics["accuracy"]), f"{report['test_patients']} preserved test patients · exploratory"),
               ("Balanced accuracy", pct(metrics["balanced_accuracy"]), "Equal weight for yes and no labels"),
               ("Test AUC", number(metrics["roc_auc"]), "0.5 is chance ranking"),
               ("Recorded recurrence recall", pct(metrics["recurrence_recall"]), f"Fraction of the {report['test_recorded_recurrences']} recorded recurrences found")])
        st.warning("This is a weak exploratory classifier. The always-no baseline has "
                   f"{report['prevalence_reference_test']['accuracy']:.1%} accuracy and 0% recurrence recall. "
                   "Compare recurrence recall and balanced accuracy alongside overall accuracy.")
        rows = [{"Model": MODEL_NAMES[model], "Development Brier ↓": report["development_metrics"][model]["brier_score"],
                 "Test accuracy": value["accuracy"], "Test balanced accuracy": value["balanced_accuracy"],
                 "Test AUC": value["roc_auc"], "Recurrence recall": value["recurrence_recall"]}
                for model, value in report["test_metrics"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Model family was selected using development probability error. The decision threshold was selected on the same development out-of-fold scores, so development threshold metrics are optimistic. Test scores remain exploratory.")
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=["Actual: no recorded recurrence", "Actual: recorded recurrence"],
                              columns=["Predicted: no recorded recurrence", "Predicted: recorded recurrence"])
        st.dataframe(matrix, width="stretch")
        for limitation in report["limitations"]:
            st.write("• " + limitation)
        st.download_button("Download recorded-recurrence experiment", report_path.read_bytes(), "recorded_recurrence_report.json", "application/json", key="recorded_experiment_export")
    return True
