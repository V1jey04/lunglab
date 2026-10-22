"""A local research workspace backed by saved models and measured inference."""
from html import escape
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .dashboard_runtime import input_signature, model_comparison, preserved_cases, stream_inference
from .predict import load_bundle
from .recurrence_predict import read_feature_csv
from .reports import classification_runs, load_report, report_file

TEAL, NAVY, AMBER = "#0b9488", "#254967", "#dbad5a"
LABELS = {"adenocarcinoma": "Adenocarcinoma", "squamous_cell_carcinoma": "Squamous cell carcinoma",
          "benign": "Benign", "malignant": "Malignant", "recorded_recurrence": "Recorded recurrence",
          "no_recorded_recurrence": "No recorded recurrence"}
NAMES = {"rich_accuracy_logistic": "Regularized logistic", "baseline_rich_logistic": "Regularized logistic",
         "bagged_rich_logistic": "Bagged logistic · experimental", "paper_mlp": "Neural network · paper baseline",
         "paper_mlp_accuracy": "Neural network · paper baseline", "catboost": "CatBoost",
         "clinical_cox": "Clinical Cox", "radiomics_cox": "Radiomics Cox",
         "prevalence_reference": "Nodal prevalence reference", "volume_logistic": "Tumour volume benchmark",
         "radiomics_logistic": "Nodal radiomics logistic"}
LUNGS = '''<svg viewBox="0 0 120 120" fill="none" aria-hidden="true">
<path d="M60 18v30m0-9-13 14M60 39l13 14" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
<path d="M47 37c-7-8-18-2-23 14-5 14-8 34-2 40 7 7 24-2 28-10 3-7 1-28-3-44ZM73 37c7-8 18-2 23 14 5 14 8 34 2 40-7 7-24-2-28-10-3-7-1-28 3-44Z" stroke="currentColor" stroke-width="2" fill="currentColor" fill-opacity=".07"/>
<path d="m46 54-12 9m13 2-14 13m41-24 12 9M73 65l14 13" stroke="currentColor" stroke-width="1.5" opacity=".5"/>
<circle cx="60" cy="60" r="55" stroke="currentColor" opacity=".12" stroke-dasharray="3 5"/>
<path d="M6 35V15h20m88 20V15H94M6 85v20h20m88-20v20H94" stroke="currentColor" opacity=".4"/></svg>'''


def pretty(name):
    return NAMES.get(name, name.replace("_", " ").capitalize())


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def pct(value):
    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}"


def number(value):
    return "—" if value is None or pd.isna(value) else f"{float(value):.3f}"


def cards(items):
    html('<div class="stats">' + ''.join(
        f'<div class="stat-card"><div class="stat-label">{escape(label)}</div><div class="stat-value">{escape(str(value))}</div>'
        f'<div class="stat-note">{escape(note)}</div><span class="stat-icon">◈</span></div>'
        for label, value, note in items) + '</div>')


def panel(title, kicker="", detail=""):
    html(f'<div class="section-kicker">{escape(kicker)}</div><div class="panel-heading"><h3>{escape(title)}</h3>'
         f'<span class="panel-detail">{escape(detail)}</span></div>')


@st.cache_resource(show_spinner=False)
def cached_bundle(path, modified_ns):
    return load_bundle(path)


def chart(frame, spec, height=260):
    spec = {**spec, "height": height, "background": "transparent", "config": {
        "view": {"stroke": None}, "axis": {"domain": False, "gridColor": "#ecf1f6", "tickColor": "#dce5ee",
        "labelColor": "#74869a", "titleColor": "#74869a", "labelFontSize": 10, "titleFontSize": 10, "labelPadding": 8},
        "legend": {"labelColor": "#6b8096", "title": None, "orient": "bottom"}}}
    st.vega_lite_chart(frame, spec, width="stretch", theme=None)


def pipeline(stage="ready", completed=0, total=0):
    active = {"ready": -1, "validated": 1, "prediction": 2, "complete": 4}.get(stage, -1)
    steps = [("Input schema", "Names & numeric values"), ("Feature pipeline", "Frozen preprocessing"),
             ("Model inference", f"{completed} / {total} cases" if total else "Awaiting a case queue"), ("Results", "Probabilities & export")]
    html('<div class="pipeline">' + ''.join(
        f'<div class="pipeline-step {"done" if i < active else "active" if i == active else ""}">'
        f'<div class="number">{"✓" if i < active else f"0{i + 1}"}</div><strong>{title}</strong><small>{escape(note)}</small></div>'
        for i, (title, note) in enumerate(steps)) + '</div>')


def live_output(result=None, patient_id=None, status="Ready"):
    head = '<div class="live-box"><div class="live-top"><span class="section-kicker">MODEL OUTPUT</span>'
    head += f'<span class="badge"><i class="status-dot"></i>{escape(status)}</span></div>'
    if result is None:
        html(head + '<div class="live-placeholder">' + LUNGS + '<p>Your next case starts here.<br>Run a queue to inspect real model outputs.</p></div>'
             '<div class="live-foot">LOCAL MODEL · NO SIMULATED SCORES</div></div>')
        return
    prediction = str(result["prediction"])
    probability, circumference = float(result[f"p_{prediction}"]), 2 * np.pi * 44
    ring = f'''<svg viewBox="0 0 110 110" role="img" aria-label="Model probability {probability:.1%}">
    <circle cx="55" cy="55" r="44" fill="none" stroke="#284158" stroke-width="7"/>
    <circle cx="55" cy="55" r="44" fill="none" stroke="#76dcc7" stroke-width="7" stroke-linecap="round"
    stroke-dasharray="{probability*circumference} {circumference}" transform="rotate(-90 55 55)"/>
    <text x="55" y="58" text-anchor="middle" fill="#e3f6f1" font-size="21" font-family="sans-serif" font-weight="600">{probability:.0%}</text>
    <text x="55" y="75" text-anchor="middle" fill="#7696ae" font-size="8" font-family="sans-serif">PROBABILITY</text></svg>'''
    body = f'<h3>{escape(str(patient_id))}</h3><div class="live-readout">{ring}<div><div class="readout-value">'
    body += escape(LABELS.get(prediction, prediction)) + '</div><div class="readout-label">Predicted class<br>Frozen model decision</div></div></div>'
    for column in [column for column in result.index if column.startswith("p_")]:
        value = float(result[column])
        body += f'<div class="probability-row"><span>{escape(LABELS.get(column[2:], column[2:]))}</span><span>{value:.1%}</span></div>'
        body += f'<div class="probability-track"><span style="width:{value*100:.2f}%"></span></div>'
    review = "EXPERIMENTAL REVIEW FLAG" if bool(result.get("review_suggested", False)) else "NO EXPERIMENTAL REVIEW FLAG"
    html(head + body + f'<div class="live-foot" style="margin-top:20px">{review}</div></div>')


def activity_log(events):
    html(''.join(f'<div class="event-row"><span><span class="event-check">✓</span>{escape(text)}</span><code>{duration:.1f} ms</code></div>'
                 for text, duration in events[-6:]))


def render_live(run, report, name):
    model_path = run / f"{name}.joblib"
    exists = model_path.is_file()
    left, right = st.columns([1.55, 1], gap="large")
    frame, bundle = None, None
    with left:
        with st.container(border=True, key="panel_queue"):
            panel("Build your case queue", "01 / LIVE INFERENCE", "CSV features → saved model")
            source = st.radio("Input source", ["Preserved test cases", "Upload feature CSV"], horizontal=True, key="input_source")
            if source == "Preserved test cases":
                examples = preserved_cases(run, report)
                if examples.empty:
                    st.info("Example features are unavailable. Upload a feature CSV to run inference.")
                else:
                    upper = min(len(examples), 50)
                    count = st.slider("Cases to run", 1, upper, min(5, upper), key=f"queue_size_{run.name}") if upper > 1 else 1
                    frame = examples.head(count).copy()
                    st.caption("Previously inspected test patients. Example inference is not a new validation result.")
            else:
                uploaded = st.file_uploader("Feature CSV", type=["csv"], key="live_feature_csv")
                if uploaded is not None:
                    try:
                        frame = read_feature_csv(uploaded)
                        if len(frame) > 500:
                            st.error("Queue at most 500 cases per live run. Use the CLI for larger exports.")
                            frame = None
                    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
                        st.error(str(error))
                st.caption("One row per patient. Match the training extraction procedure and required feature names.")
            if exists:
                try:
                    bundle = cached_bundle(str(model_path.resolve()), model_path.stat().st_mtime_ns)
                except Exception as error:
                    st.error(f"Unable to load the saved model: {error}")
            else:
                st.info("This report has no model bundle available for live inference.")
            if frame is not None:
                st.dataframe(frame.loc[:, [x for x in ("patient_id", "sample_id") if x in frame]], hide_index=True,
                             width="stretch", height=min(180, 36 * len(frame) + 38))
            go = st.button("Run live inference  →", type="primary", width="stretch", key="run_live", disabled=frame is None or bundle is None)
        with st.container(border=True, key="panel_monitor"):
            panel("Execution monitor", "02 / MEASURED ACTIVITY", "Updates as each case finishes")
            steps_slot, progress_slot, logs_slot = st.empty(), st.empty(), st.empty()
    with right:
        output_slot = st.empty()
        with st.container(border=True, key="panel_explanation"):
            panel("Read the result", "INTERPRETATION")
            html('<div class="explain-title">Probability is a model output</div><div class="explain-copy">It describes the relative likelihood '
                 'of the supported classes. It is not the model’s accuracy or a validated measure of clinical certainty.</div>')
            rule = (bundle or {}).get("prediction_rule", "probability_threshold")
            decision = "The fitted classifier’s native decision determines the class." if rule == "native_predict" else (
                f'The positive class is selected at a frozen probability threshold of {float((bundle or {}).get("threshold", .5)):.2f}.')
            html(f'<div class="explain-title">How the class is selected</div><div class="explain-copy">{escape(decision)}</div>'
                 '<div class="explain-title">Why a case gets a review flag</div><div class="explain-copy">The selected-class probability is below 70%, '
                 'or the positive-class probability is within 0.10 of the saved threshold. This flag is experimental.</div>')
    signature = input_signature(model_path, frame) if frame is not None and exists else None
    saved = st.session_state.get("live_run")
    if saved and saved["signature"] != signature:
        saved = None
        st.session_state.pop("live_run", None)
    if go:
        saved, events = None, []
        st.session_state.pop("live_run", None)
        try:
            for event in stream_inference(bundle, frame):
                with steps_slot.container():
                    pipeline(event.stage, event.completed, event.total)
                progress_slot.progress(event.completed / event.total,
                    text=f"{event.completed} / {event.total} cases complete · {event.compute_ms:.1f} ms model time")
                if event.stage == "validated":
                    events.append((f"Validated {event.total} cases · {len(bundle['feature_names']):,} inputs", event.elapsed_ms))
                elif event.stage == "prediction":
                    events.append((f"{event.patient_id} · inference complete", event.case_ms))
                    with output_slot.container():
                        live_output(event.result.iloc[0], event.patient_id, "Processing")
                else:
                    events.append(("Batch complete · wall time", event.elapsed_ms))
                    saved = {"signature": signature, "result": event.result, "compute_ms": event.compute_ms,
                             "elapsed_ms": event.elapsed_ms, "events": events}
                    st.session_state["live_run"] = saved
                with logs_slot.container():
                    activity_log(events)
        except Exception as error:
            progress_slot.empty()
            with steps_slot.container():
                pipeline()
            with logs_slot.container():
                st.error(f"Inference stopped: {error}")
    if saved:
        result = saved["result"]
        with right:
            patient = st.selectbox("Inspect a completed case", result.patient_id.astype(str).tolist(), key="inspect_case")
        with output_slot.container():
            live_output(result.loc[result.patient_id.astype(str).eq(patient)].iloc[0], patient, "Complete")
        with steps_slot.container():
            pipeline("complete", len(result), len(result))
        progress_slot.progress(1.0, text=f"{len(result)} / {len(result)} cases complete · {saved['compute_ms']:.1f} ms model time")
        with logs_slot.container():
            activity_log(saved["events"])
        a, b, c = st.columns(3)
        a.metric("Cases completed", len(result))
        b.metric("Measured model time", f"{saved['compute_ms']:.1f} ms", help="Actual prediction calls, excluding interface rendering.")
        c.metric("Review flags", int(result.review_suggested.sum()))
        with st.container(border=True, key="panel_results"):
            panel("Batch results", "03 / CASE EXPLORER", "Inspect, sort & download")
            st.dataframe(result, hide_index=True, width="stretch", column_config={column: st.column_config.ProgressColumn(
                LABELS.get(column[2:], column[2:]), format="%.3f", min_value=0, max_value=1) for column in result if column.startswith("p_")})
            st.download_button("Download prediction CSV", result.to_csv(index=False), "lunglab_predictions.csv", "text/csv")
            positive = report["positive_class"]
            plot = result.assign(Case=np.arange(1, len(result) + 1))
            chart(plot, {"mark": {"type": "circle", "size": 100}, "params": [{"name": "zoom", "select": "interval", "bind": "scales"}],
                "encoding": {"x": {"field": "Case", "type": "quantitative", "axis": {"tickMinStep": 1}},
                "y": {"field": f"p_{positive}", "type": "quantitative", "title": f"P({LABELS.get(positive, positive)})",
                      "scale": {"domain": [0, 1]}, "axis": {"format": ".0%"}},
                "color": {"field": "review_suggested", "type": "nominal", "title": "Review flag",
                          "scale": {"domain": [False, True], "range": [TEAL, AMBER]}},
                "tooltip": [{"field": "patient_id"}, {"field": "prediction"}, {"field": f"p_{positive}", "format": ".1%"}]}}, 190)
            st.caption("Scroll or drag to explore the batch. Inference probabilities are not new performance estimates.")
        from .dashboard_inspection import render_case_inspection
        selected_frame = frame.loc[frame.patient_id.astype(str).eq(patient)].copy()
        selected_result = result.loc[result.patient_id.astype(str).eq(patient)].copy()
        render_case_inspection(model_path, bundle, selected_frame, selected_result, key="live_case")
    else:
        with output_slot.container():
            live_output()
        with steps_slot.container():
            pipeline()
        if not go:
            with logs_slot.container():
                html('<div class="empty-state">Queue ready. Start inference to see measured case timings and model outputs.</div>')


def render_comparison(run, report, selected):
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        with st.container(border=True, key="panel_comparison"):
            panel("Performance across models", "FROZEN EXPERIMENT", "Development selection · held-out evaluation")
            metric = st.radio("Comparison metric", ["Test accuracy", "Balanced accuracy", "Test AUC", "Development AUC"], horizontal=True)
            table = model_comparison(report)
            table["Display model"] = table.Model.map(pretty)
            chart(table, {"mark": {"type": "bar", "cornerRadiusEnd": 5, "height": 19}, "encoding": {
                "y": {"field": "Display model", "type": "nominal", "sort": "-x", "title": None, "axis": {"labelLimit": 240}},
                "x": {"field": metric, "type": "quantitative", "scale": {"domain": [0, 1]}, "axis": {"format": ".0%"}, "title": None},
                "color": {"field": "Selected", "type": "nominal", "scale": {"domain": [False, True], "range": ["#bdccdd", TEAL]}, "legend": None},
                "tooltip": [{"field": "Display model"}, {"field": metric, "format": ".3f"}, {"field": "Selected"}]}}, max(180, len(table) * 42))
            st.caption("Teal marks the development-selected model. Exploring another candidate does not change selection.")
            st.dataframe(table.drop(columns="Display model"), hide_index=True, width="stretch")
            st.download_button("Download experiment report", report_file(run).read_bytes(), "lunglab_report.json", "application/json")
    with right:
        with st.container(border=True, key="panel_confusion"):
            panel("Where the model is right", "CONFUSION MATRIX", pretty(selected))
            value = report["models"][selected]
            metrics = value.get("test_metrics", {})
            matrix, labels = metrics.get("confusion_matrix"), report["labels"]
            if matrix is not None:
                cells = pd.DataFrame([{"Actual": LABELS.get(labels[i], labels[i]), "Predicted": LABELS.get(labels[j], labels[j]), "Patients": count}
                                      for i, row in enumerate(matrix) for j, count in enumerate(row)])
                chart(cells, {"encoding": {"x": {"field": "Predicted", "type": "nominal", "axis": {"labelAngle": 0, "labelLimit": 120}},
                                           "y": {"field": "Actual", "type": "nominal", "axis": {"labelLimit": 120}}},
                    "layer": [{"mark": {"type": "rect", "cornerRadius": 5}, "encoding": {
                        "color": {"field": "Patients", "type": "quantitative", "scale": {"range": ["#edf6f4", "#188f83"]}, "legend": None},
                        "tooltip": [{"field": "Actual"}, {"field": "Predicted"}, {"field": "Patients"}]}},
                        {"mark": {"type": "text", "fontSize": 24, "fontWeight": 600, "color": "#123d3b"},
                         "encoding": {"text": {"field": "Patients"}}}]}, 210)
            st.caption("Diagonal cells are correct predictions; off-diagonal cells are errors on preserved test patients.")
            a, b = st.columns(2)
            a.metric("Sensitivity", pct(metrics.get("sensitivity")))
            b.metric("Specificity", pct(metrics.get("specificity")))
            intervals = value.get("test_intervals", value.get("test_ci95", {}))
            if "accuracy" in intervals:
                low, high = intervals["accuracy"]
                st.caption(f"Accuracy 95% bootstrap interval: {low:.1%}–{high:.1%}.")
            html('<div class="explain-copy">Balanced accuracy gives equal weight to both class recalls. AUC measures ranking across thresholds. '
                 'Neither establishes usefulness in a new clinical population.</div>')
    if report.get("previously_inspected_holdout"):
        st.info("Exploratory evidence: these test patients were inspected in earlier experiments. Independent patients are needed for confirmation.")
    with st.expander("Global feature associations & saved diagnostic plots"):
        path = run / f"{selected}_feature_importance.csv"
        if path.exists():
            st.bar_chart(pd.read_csv(path).head(12).set_index("feature"), horizontal=True)
            st.caption("Global training-model associations; not an individual explanation or a causal mechanism.")
        else:
            st.caption("This bundle does not include feature contribution values. No individual attribution is inferred from its probabilities.")
        if (run / "comparison.png").exists():
            st.image(str(run / "comparison.png"), width="stretch")


def render_extension_status():
    """Make data readiness distinct from the presence of prediction code."""
    rows = []
    for label, directory in [("Lymph-node involvement", "nodal"), ("Time-based recurrence", "recurrence")]:
        audit_path = Path(f"artifacts/{directory}_audit/audit.json")
        selection = Path(f"artifacts/{directory}/selection.json")
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        rows.append({"Extension": label,
                     "Cohort": "Prepared" if audit else "Awaiting preparation",
                     "Training data": "Definitions verified" if audit.get("training_allowed") else "Verification required",
                     "Model": "Frozen experiment present" if selection.exists() else "No trained model"})
    recorded = Path("artifacts/recurrence_recorded/report.json")
    if recorded.is_file():
        recorded_report = json.loads(recorded.read_text())
        total = recorded_report["development_patients"] + recorded_report["test_patients"]
        rows.append({"Extension": "Recorded recurrence classifier", "Cohort": f"{total} labelled patients",
                     "Training data": "Variable follow-up · exploratory", "Model": "Trained · open Recurrence to predict"})
    with st.expander("Clinical extension readiness", expanded=True):
        for column, row in zip(st.columns(len(rows)), rows):
            with column:
                st.markdown(f"**{row['Extension']}**")
                st.caption(f"{row['Cohort']} · {row['Training data']}")
                st.write(row["Model"])
        st.caption("The trained classifier predicts source-recorded recurrence. Fixed-horizon recurrence and nodal models retain their separate verification requirements.")


def render_nodal():
    audit_path, output = Path("artifacts/nodal_audit/audit.json"), Path("artifacts/nodal")
    selection_path = output / "selection.json"
    panel("Estimate lymph-node involvement", "SPREAD AT DIAGNOSIS", "Pathological N0 versus N1/N2/N3")
    st.caption("Research goal: use a matching preoperative CT to estimate whether cancer involves lymph nodes. Patient-level labels do not locate individual affected nodes.")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        counts = audit["counts"]
        cards([("Source patients", counts["source_patients"], "Patients with tumour feature rows"),
               ("Recorded node-positive", counts["source_node_positive"], "Observed pathology, not model predictions"),
               ("Provisional eligible", counts["eligible_patients"], "CT-to-surgery window: 1–90 days"),
               ("Eligible node-positive", counts["eligible_node_positive"], "Small minority limits model evidence")])
        if not audit["training_allowed"]:
            st.warning("Nodal training awaits a verified CT-to-surgery interpretation and radiomic scan match. No nodal probability is available yet.")
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            with st.container(border=True, key="panel_nodal_labels"):
                panel("Recorded nodal labels", "PROVISIONAL COHORT")
                cases = pd.DataFrame([{"Split": split.title(), "Label": label.replace("_", " ").title(), "Patients": count}
                                      for split, values in counts["eligible_by_split"].items()
                                      for label, count in values.items()])
                chart(cases, {"mark": {"type": "bar", "cornerRadiusEnd": 5}, "encoding": {
                    "y": {"field": "Split", "type": "nominal", "title": None},
                    "x": {"field": "Patients", "type": "quantitative", "stack": "zero"},
                    "color": {"field": "Label", "type": "nominal", "scale": {"range": [NAVY, TEAL]}},
                    "tooltip": [{"field": "Split"}, {"field": "Label"}, {"field": "Patients"}]}}, 160)
                st.caption("Pathology supplies the training target. Histology, stage, treatment and recurrence fields are excluded from predictors.")
        with right:
            with st.container(border=True, key="panel_nodal_audit"):
                panel("Evidence needed before training", "DATA READINESS")
                for value in audit.get("blocking_reason", {}).values():
                    st.write(value)
                if audit["training_allowed"]:
                    st.success("Scan timing evidence is recorded. Nested model selection can run when class support is sufficient.")
                st.download_button("Download nodal audit", audit_path.read_bytes(), "nodal_audit.json", "application/json")
        with st.expander("Nodal cohort limits and exclusions"):
            for limitation in audit.get("limitations", []):
                st.write(limitation)
            st.dataframe(pd.DataFrame([{"Reason": reason.replace("_", " "), "Patients": count}
                                      for reason, count in counts["exclusions_by_reason"].items()]),
                         hide_index=True, width="stretch")
            st.caption("Exclusion reasons can overlap. Clinical N stage is unavailable, so this is not an established occult-spread model.")
    else:
        st.info("Prepare nodal labels with: .venv/bin/lung-ml nodal prepare")
    if not selection_path.exists():
        st.session_state.pop("nodal_run", None)
        return
    selection = json.loads(selection_path.read_text())
    from .nodal_train import FAMILIES, verify_frozen_models
    try:
        verify_frozen_models(output, selection)
    except ValueError as error:
        st.session_state.pop("nodal_run", None)
        st.error(str(error))
        return
    st.subheader("Frozen nodal experiment")
    st.caption(f"Selected by nested development Brier score: {pretty(selection['selected_model_by_development'])}.")
    comparison = [{"Model": pretty(family), "Nested Brier": values["nested_metrics"]["brier_score"],
                   "Nested AUC": values["nested_metrics"]["roc_auc"],
                   "Node-positive recall": values["nested_metrics"]["node_positive_recall"]}
                  for family, values in selection["development_metrics"].items()]
    st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
    report_path = output / "external_report.json"
    if report_path.exists():
        with st.expander("Exploratory nodal test scores"):
            report = json.loads(report_path.read_text())
            st.write(report["evaluation"])
            st.json(report["test_metrics"])
        st.download_button("Download nodal report", report_path.read_bytes(), "nodal_report.json", "application/json")
    name = st.selectbox("Nodal model", FAMILIES, index=FAMILIES.index(selection["selected_model_by_development"]),
                        format_func=pretty, key="nodal_model")
    uploaded = st.file_uploader("Nodal feature CSV", type=["csv"], key="nodal_csv")
    example = st.checkbox("Use five preserved nodal test cases", key="nodal_example")
    frame = None
    if uploaded is not None:
        try:
            frame = read_feature_csv(uploaded)
        except Exception as error:
            st.error(str(error))
    elif example:
        source = Path(selection["dataset_path"])
        if source.is_file():
            frame = pd.read_csv(source, dtype={"patient_id": str, "sample_id": str})
            frame = frame.loc[frame.patient_id.isin(selection["test_patient_ids_excluded_from_selection"])].head(5)
            st.caption("Previously inspected patients; example inference is not new validation.")
    if frame is None or frame.empty or len(frame) > 500:
        st.session_state.pop("nodal_run", None)
        if frame is not None:
            st.error("Queue between 1 and 500 patients.")
        return
    path = output / f"{name}.joblib"
    signature = input_signature(path, frame)
    state = st.session_state.get("nodal_run")
    if state and state["signature"] != signature:
        state = None
        st.session_state.pop("nodal_run", None)
    if st.button("Run nodal inference", type="primary", key="run_nodal"):
        state = None
        st.session_state.pop("nodal_run", None)
        try:
            progress = st.progress(0.0)
            for event in stream_inference(joblib.load(path), frame):
                progress.progress(event.completed / event.total, text=f"{event.completed} / {event.total} cases complete")
                if event.stage == "complete":
                    state = {"signature": signature, "result": event.result, "compute_ms": event.compute_ms}
                    st.session_state["nodal_run"] = state
        except Exception as error:
            st.error(str(error))
    if state:
        result = state["result"]
        st.caption(f"Measured model time: {state['compute_ms']:.1f} ms. Individual confidence intervals have not been estimated.")
        patient = st.selectbox("Inspect nodal case", result.patient_id.unique(), key="nodal_patient")
        row = result.loc[result.patient_id.eq(patient)].iloc[0]
        a, b = st.columns(2)
        a.metric("Estimated node-positive probability", pct(row.p_node_positive))
        b.metric("Frozen decision threshold", pct(row.decision_threshold))
        st.caption("A probability estimate does not confirm staging. The threshold is a research operating point, not a biopsy or treatment rule.")
        st.dataframe(result, hide_index=True, width="stretch")
        st.download_button("Download nodal estimates", result.to_csv(index=False), "nodal_predictions.csv", "text/csv")


def render_recurrence():
    from .dashboard_recurrence import render_recorded_recurrence
    if render_recorded_recurrence():
        with st.expander("Time-based recurrence research · source verification pending"):
            render_timed_recurrence()
    else:
        render_timed_recurrence()


def render_timed_recurrence():
    audit_path, output = Path("artifacts/recurrence_audit/audit.json"), Path("artifacts/recurrence")
    selection_path = output / "selection.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        counts = audit["counts"]
        panel("Understanding recurrence readiness", "RECURRENCE RESEARCH", "First recurrence · competing death")
        cards([("Provisional eligible patients", counts["eligible_by_dates_and_clinical_rules"], "Surgery-date interpretation remains provisional"),
               ("Recorded first recurrences", counts["eligible_recurrence_events"], "Observed events, not predictions"),
               ("Competing deaths", counts["eligible_competing_deaths"], "Death before recorded recurrence"),
               ("Right-censored patients", counts["eligible_right_censored"], "Later recurrence status is unobserved")])
        if not audit["training_allowed"]:
            html('<div class="note-card"><strong>Awaiting verified outcome definitions.</strong> Recurrence prediction is locked until the '
                 'scan-to-surgery match and recurrence assessment through follow-up are documented. No recurrence probability is available from the current clinical labels.</div>')
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            with st.container(border=True, key="panel_observed"):
                panel("What happened in the source cohort", "OBSERVED FIRST EVENTS")
                names = {"recurrence": "Recurrence", "competing_death": "Competing death", "censored": "Censored"}
                events = pd.DataFrame([{"Split": split.title(), "Event": names[key], "Patients": value}
                                       for split, values in counts["event_counts_by_split"].items() for key, value in values.items()])
                chart(events, {"mark": {"type": "bar", "cornerRadiusEnd": 5, "height": 30}, "encoding": {
                    "y": {"field": "Split", "type": "nominal", "title": None}, "x": {"field": "Patients", "type": "quantitative", "stack": "zero"},
                    "color": {"field": "Event", "type": "nominal", "scale": {"domain": list(names.values()), "range": [TEAL, NAVY, "#cedae7"]}},
                    "tooltip": [{"field": "Split"}, {"field": "Event"}, {"field": "Patients"}]}}, 150)
                horizon = st.radio("Observation horizon", [365, 730], format_func=lambda x: f"{x} days", horizontal=True, key="audit_horizon")
                followup = counts["days_followed_at_horizon_by_split"]
                a, b = st.columns(2)
                a.metric("Development followed ≥ horizon", followup["development"][str(horizon)])
                b.metric("Test followed ≥ horizon", followup["test"][str(horizon)])
                st.caption("Recorded durations do not establish verified recurrence surveillance.")
        with right:
            with st.container(border=True, key="panel_audit"):
                panel("Why fitting is locked", "DATA AUDIT")
                for key, value in audit.get("blocking_reason", {}).items():
                    html(f'<div class="explain-title">{escape(key.replace("_", " ").capitalize())}</div><div class="explain-copy">{escape(value)}</div>')
                st.download_button("Download cohort audit", audit_path.read_bytes(), "recurrence_audit.json", "application/json")
        with st.expander("Exclusions and cohort construction"):
            st.dataframe(pd.DataFrame([{"Reason": key.replace("_", " "), "Patients": value}
                for key, value in counts["exclusions_by_reason"].items()]), hide_index=True, width="stretch")
            st.caption("Reasons overlap. The primary preoperative CT window is 1–90 days.")
    elif not selection_path.exists():
        st.info("Prepare the recurrence cohort to explore outcome readiness and follow-up support.")
    if not selection_path.exists():
        return
    selection = json.loads(selection_path.read_text())
    st.subheader("Frozen recurrence experiment")
    st.caption(f"Selected using nested development evidence: {pretty(selection['selected_model_by_development'])}.")
    rows = [{"Model": pretty(family), "Horizon (days)": int(horizon), "Nested IPCW Brier": score,
             "Competing-risk AUC": metrics.get("nested_competing_risk_auc_by_horizon", {}).get(horizon)}
            for family, metrics in selection.get("development_metrics", {}).items()
            for horizon, score in metrics.get("nested_ipcw_brier_by_horizon", {}).items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    external = output / "external_report.json"
    if external.exists():
        report = json.loads(external.read_text())
        st.info(report["interpretation"])
        with st.expander("Exploratory test scores"):
            st.json(report["test_metrics"])
        st.download_button("Download recurrence report", external.read_bytes(), "recurrence_report.json", "application/json")
    paths = [output / f"{family}.joblib" for family in ("clinical_cox", "radiomics_cox") if (output / f"{family}.joblib").exists()]
    if not paths:
        return
    from .recurrence_train import _verify_frozen_models
    try:
        _verify_frozen_models(output, selection)
    except ValueError as error:
        st.error(str(error))
        return
    name = st.selectbox("Recurrence model", [path.stem for path in paths], format_func=pretty)
    uploaded = st.file_uploader("Recurrence feature CSV", type=["csv"], key="recurrence_csv")
    example = st.checkbox("Use five preserved test feature rows as an inference example", key="recurrence_example")
    frame = None
    if uploaded is not None:
        try:
            frame = read_feature_csv(uploaded)
        except Exception as error:
            st.error(str(error))
    elif example:
        features, splits = Path("data/recurrence_features.csv"), Path("data/recurrence_cohort.splits.csv")
        if features.exists() and splits.exists():
            frame = pd.read_csv(features, dtype={"patient_id": str})
            manifest = pd.read_csv(splits, dtype={"patient_id": str})
            frame = frame.loc[frame.patient_id.isin(manifest.loc[manifest.split.eq("test"), "patient_id"])].head(5)
            st.caption("Previously inspected test patients; example inference is not new validation.")
    if frame is None:
        return
    if len(frame) > 500:
        st.error("Queue at most 500 cases per live run.")
        return
    path = output / f"{name}.joblib"
    signature = input_signature(path, frame)
    state = st.session_state.get("recurrence_run")
    if state and state["signature"] != signature:
        state = None
        st.session_state.pop("recurrence_run", None)
    if st.button("Run recurrence inference", type="primary"):
        state = None
        st.session_state.pop("recurrence_run", None)
        try:
            progress = st.progress(0.0)
            for event in stream_inference(joblib.load(path), frame):
                progress.progress(event.completed / event.total, text=f"{event.completed} / {event.total} cases complete")
                if event.stage == "complete":
                    state = {"signature": signature, "result": event.result, "compute_ms": event.compute_ms}
                    st.session_state["recurrence_run"] = state
        except Exception as error:
            st.error(str(error))
    if state:
        result = state["result"]
        st.caption(f"Measured model time: {state['compute_ms']:.1f} ms. Estimates describe represented study care, not treatment effects.")
        patient = st.selectbox("Inspect recurrence case", result.patient_id.unique())
        curve = result.loc[result.patient_id.eq(patient)].set_index("horizon_days_after_index")[[
            "recurrence_cumulative_incidence", "competing_death_cumulative_incidence", "event_free_probability"]]
        st.line_chart(curve.rename(columns={"recurrence_cumulative_incidence": "Recurrence", "competing_death_cumulative_incidence": "Competing death", "event_free_probability": "Event-free"}))
        st.caption("Lines connect supported horizon estimates; they do not imply observations between horizons.")
        st.dataframe(result, hide_index=True, width="stretch")
        st.download_button("Download recurrence estimates", result.to_csv(index=False), "recurrence_predictions.csv", "text/csv")


def render_methods(run, report):
    panel("A transparent path from inputs to evidence", "METHODS & DATA")
    columns = st.columns(3)
    for column, title, copy in zip(columns, ["01 · Numerical tumour features", "02 · Frozen preprocessing", "03 · Patient-level evidence"], [
        "Radiomics measures tumour shape, CT intensity and spatial texture within a known segmentation. Match the model’s extraction procedure.",
        "Imputation, scaling and feature selection are learned inside training folds. Saved bundles apply those fitted transforms to each new case.",
        "Development data guide model choice. Test patients are evaluated after selection. Previously inspected cohorts remain exploratory."]):
        with column:
            with st.container(border=True, key=f"panel_method_{title}"):
                html(f'<div class="explain-title">{title}</div><div class="explain-copy">{copy}</div>')
    st.info("Research prototype: classification assumes a known segmented tumour. It does not detect or localize an unknown cancer in a CT scan. Outputs are not treatment recommendations.")
    with st.expander("Metric glossary", expanded=True):
        st.markdown("**Accuracy** is the proportion of correct classifications. **Balanced accuracy** averages the two class recalls. "
            "**AUC** measures positive-class ranking across thresholds. **Cumulative incidence** accounts for competing death when estimating recurrence probability. "
            "The **IPCW Brier score** measures probability error with censoring weights; lower values are better.")
    if report:
        with st.expander("Selected experiment & provenance"):
            st.write(report["interpretation"])
            st.write(report.get("evaluation", "Patient-level development/test split."))
            st.code(f"Experiment: {run.name}\nSelected model: {report['selected_model_by_cv']}\nSource: {report['dataset_path']}", language=None)
            provenance = Path(report["dataset_path"]).with_suffix(".provenance.json")
            if provenance.exists():
                st.json(json.loads(provenance.read_text()))
    with st.expander("Create features from CT volumes"):
        st.caption("Use CT volumes and matching tumour masks. JPG/PNG screenshots do not provide Hounsfield units or 3D geometry.")
        st.code(".venv/bin/lung-ml extract --manifest data/manifest.csv --output data/extracted_features.csv\n"
                ".venv/bin/lung-ml train --data data/extracted_features.csv --output artifacts/my_ct_run", language="bash")


def render_experiments():
    panel("The research notebook", "DEVELOPMENT EXPERIMENTS", "Selection evidence, preserved for review")
    st.caption("Development experiments guide model choices. Results need independent validation.")
    paths = [Path("artifacts/improvement_development/search_report.json"), *[Path("artifacts") / name / "selection.json" for name in (
        "radiogenomics_rich", "radiogenomics_accuracy", "radiogenomics_expanded_clinical", "radiogenomics_refinement")],
        Path("artifacts/foundation_development/development_report.json")]
    for path in paths:
        if not path.exists():
            continue
        experiment = json.loads(path.read_text())
        with st.expander(pretty(path.parent.name)):
            rows = []
            for name, value in experiment.get("models", {}).items():
                m = value.get("metrics", value)
                rows.append({"Model": pretty(name), "CV accuracy": m.get("accuracy", m.get("nested_cv_accuracy")),
                    "CV balanced accuracy": m.get("balanced_accuracy", m.get("nested_cv_balanced_accuracy")),
                    "CV AUC": m.get("roc_auc", m.get("nested_cv_auc", m.get("nested_cv_roc_auc")))})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if experiment.get("interpretation"):
                st.caption(experiment["interpretation"])
            st.download_button("Download development record", path.read_bytes(), f"{path.parent.name}_development.json", "application/json", key=f"download_{path.parent.name}")


def render_dashboard():
    st.set_page_config(page_title="LungLab · Research workspace", page_icon="🫁", layout="wide")
    html('<style>' + (Path(__file__).with_name("assets") / "dashboard.css").read_text() + '</style>')
    runs = classification_runs()
    run, report, name = None, None, None
    with st.sidebar:
        html('<div class="brand"><div class="brand-icon">' + LUNGS + '</div><div><div class="brand-name">LungLab<span style="color:#61cbb5">.</span></div>'
             '<div class="brand-sub">Radiomics workspace</div></div></div>')
        st.caption("EXPERIMENT CONTROL")
        if runs:
            preferred = Path("artifacts/radiogenomics_accuracy")
            refinement = Path("artifacts/radiogenomics_refinement")
            if (refinement / "external_report.json").exists() and json.loads((refinement / "external_report.json").read_text()).get("refinement_selected_as_default"):
                preferred = refinement
            run = st.selectbox("Experiment", runs, format_func=lambda p: p.name, index=runs.index(preferred) if preferred in runs else 0)
            report = load_report(run)
            name = st.selectbox("Model", list(report["models"]), format_func=pretty, index=list(report["models"]).index(report["selected_model_by_cv"]))
            html(f'<div class="sidebar-note"><strong>{escape(pretty(report["selected_model_by_cv"]))}</strong>'
                 'Selected using development evidence.<br>All saved candidates remain inspectable.</div>')
        else:
            st.caption("No completed classification reports yet.")
        st.divider()
        html('<div class="sidebar-note"><strong>Runs on your machine</strong>Saved models perform inference locally. No remote inference service is used by this dashboard.</div>')
        st.caption("RESEARCH USE ONLY · Known, segmented tumours")
    ready = bool(run and (run / f"{name}.joblib").is_file())
    html('<div class="topline"><div class="trail">Workspace <span style="padding:0 10px">/</span> <b>Model intelligence</b></div><span class="badge '
         + ('ready' if ready else 'audit') + '"><i class="status-dot"></i>' + ('Local model ready' if ready else 'Research workspace') + '</span></div>')
    feature_count = len(report.get("feature_names", [])) if report else 0
    html('<div class="hero"><div><div class="eyebrow">CT RADIOMICS / RESEARCH INTELLIGENCE</div><h1>From tumour features<br>to <span class="accent">measurable insight.</span></h1>'
         '<p>Run saved models, follow each case as it completes, and explore the evidence behind the numbers. A transparent workspace for lung cancer research.</p></div>'
         '<div class="hero-side"><div class="scan-mark">' + LUNGS + f'</div><div class="hero-caption"><strong>{feature_count:,}</strong>available input features<br>'
         f'<strong style="margin-top:15px">{len(report["models"]) if report else 0}</strong>models in this experiment</div></div></div>')
    if report:
        value = report["models"][name]
        metrics = value.get("test_metrics", {})
        task_label = "Subtype" if report["task"] == "subtype" else "Malignancy"
        cards([(f"{task_label} test accuracy", pct(metrics.get("accuracy")), f"{report['test_patients']} preserved test patients · exploratory"),
            (f"{task_label} balanced accuracy", pct(metrics.get("balanced_accuracy")), "Equal weight for both class recalls"),
            (f"{task_label} development AUC", number(value.get("cv_selection_auc")), "Training-fold model selection evidence"),
            (f"{task_label} development patients", report["development_patients"], "Separate from the preserved test partition")])
    render_extension_status()
    live, compare, ct, nodal, recurrence, methods, experiments = st.tabs(["Live inference", "Model comparison", "CT & mask", "Lymph-node spread", "Recurrence", "Methods & data", "Development experiments"])
    with live:
        if report:
            if report.get("data_provenance") == "synthetic":
                st.info("Synthetic demonstration model. Outputs are not cancer detection evidence.")
            render_live(run, report, name)
        else:
            st.info("Train a classification model to run live inference. The recurrence audit has its own view.")
    with compare:
        if report:
            render_comparison(run, report, name)
    with ct:
        from .dashboard_inspection import render_ct_workspace
        try:
            render_ct_workspace()
        except (ImportError, RuntimeError) as error:
            st.info(f"CT workbench requires the imaging dependencies: {error}")
    with recurrence:
        render_recurrence()
    with nodal:
        render_nodal()
    with methods:
        render_methods(run, report)
    with experiments:
        render_experiments()
    html('<div class="footer"><span><strong>LungLab</strong> · Local research workspace</span><span>Known tumour inputs · Exploratory evidence · No treatment recommendations</span></div>')
