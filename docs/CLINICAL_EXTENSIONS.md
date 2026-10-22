# Lymph-node involvement and recurrence extensions

Updated 12 October 2026. The recorded-recurrence classifier is fitted on the
public cohort and available in the Recurrence tab. The separate nodal and
time-based protocols retain their data verification gates. Software tests
use synthetic fixtures and do not establish clinical performance.

| Task | Prediction target | Current data | Training requirement |
| --- | --- | --- | --- |
| Recorded recurrence classifier | Source-recorded yes/no recurrence across varying follow-up; no time horizon | 139 patients: 37 recorded recurrences; 104 development and 35 test | Trained clinical postoperative model and optional CT candidate. Independent validation and verified surveillance remain necessary for clinical claims. |
| Lymph-node involvement | Pathological N0 versus N1/N2/N3, estimated from a matching preoperative CT | 139 source patients: 113 N0 and 26 node-positive; 120 provisionally eligible within 1–90 days of surgery, including 24 node-positive patients | Document the matching radiomic scan and the CT-to-surgery interval direction |
| Post-surgical recurrence | First recurrence before competing death at supported 365- and 730-day horizons after surgery | 115 provisional patients: 26 first recurrences, 24 competing deaths and 65 right-censored patients | Verify scan/surgery timing and recurrence assessment through each censoring date |

Nodal labels describe cancer extent at surgery. Recurrence describes a later
event. The endpoints must not share labels, evaluation metrics or probability
columns. Neither task detects an unknown tumour or recommends treatment.

## Prepare and inspect the current data

Run each command from the project directory, `/Users/vijey/Documents/ml_project_new`.

```bash
.venv/bin/lung-ml nodal prepare
```

This creates `data/nodal_cohort.csv`, separate feature/outcome tables, the
preserved split, hash-recorded provenance and `artifacts/nodal_audit/audit.json`.
The 19 excluded patients have CT-to-surgery intervals outside the provisional
1–90 day window. The provisional nodal split is 88 development patients
(17 node-positive) and 32 test patients (7 node-positive).

```bash
.venv/bin/lung-ml recurrence prepare
```

This refreshes the recurrence audit and separate feature/outcome tables. It
does not fit a model. Read `artifacts/recurrence_audit/audit.json` for per-case
exclusions, event counts and follow-up support.

Evidence options record independently established source definitions. They
are not switches to bypass an unresolved audit. For nodal preparation,
`--surgery-date-evidence` must identify a documented interpretation of the
interval and establish that the radiomic scan is the matching preoperative
CT. Recurrence preparation additionally requires
`--recurrence-followup-evidence`. Do not treat last-known-alive as a verified
recurrence-assessment date without supporting evidence.

## Nodal model protocol

The packaged protocol is `lungml/configs/nodal.yaml`. It compares a constant
prevalence reference, a tumour-volume logistic model and an original-radiomics
logistic model. The last model uses training-fold median imputation,
correlation pruning, 5 or 10 selected features, scaling and regularized
logistic regression. Histology, pathological stage, treatment, molecular
status and future outcomes are excluded from predictors. Pathological N stage
provides only the target.

Three outer development folds measure performance. Three inner folds choose
regularization and feature count using probability error (Brier score).
Each outer-fold decision threshold comes from inner out-of-fold predictions.
It maximizes balanced accuracy and is a research operating point, not a
clinical intervention rule. Final model-family selection uses nested
development Brier score, with simpler models winning exact ties.

Report node-positive and node-negative recall, their Wilson confidence
intervals, balanced accuracy, ROC-AUC, average precision, probability error
and the confusion matrix. Model probabilities do not establish calibration
on new patients. No individual prediction intervals are implemented.

After the audit is genuinely resolved, train a frozen experiment:

```bash
.venv/bin/lung-ml nodal train
```

The output includes all three fitted models, development predictions,
parameters, thresholds, patient IDs and model/cohort/split/provenance hashes.
Existing frozen output directories cannot be overwritten.

Evaluate the frozen experiment once:

```bash
.venv/bin/lung-ml nodal evaluate
```

The retained test patients were already inspected in subtype experiments.
This is an exploratory evaluation, not independent confirmation. Evaluation
checks all model hashes and patient separation before loading or scoring.

Predict from a compatible feature CSV:

```bash
.venv/bin/lung-ml nodal predict \
  --model artifacts/nodal/radiomics_logistic.joblib \
  --data data/nodal_features.csv \
  --output nodal_predictions.csv
```

Inference preserves string patient IDs and input order. It rejects ambiguous
CSV headers, duplicate patients, invalid numeric values and cases with more
than 20% missing required features. Outputs include both probabilities, the
frozen threshold, model family and input missingness. Running the full feature
table demonstrates inference; it is not a new validation result.

## Recurrence model protocol

The existing cause-specific ridge Cox pair estimates recurrence and competing
death jointly. Their cumulative incidences and event-free probability sum
to one. Clinical and radiomics models are compared with an Aalen–Johansen
reference. Supported horizons depend on development censoring information.
Selection, metrics and frozen inference retain the existing protocol in
`lungml/configs/recurrence.yaml`.

Once the audit is resolved:

```bash
.venv/bin/lung-ml recurrence train
```

Use `--evaluate` to score the preserved test patients after selection freezes.
See [the recurrence implementation plan](RECURRENCE_IMPLEMENTATION_PLAN.md)
for event definitions, competing-risk metrics and support checks.

## Dashboard and research claims

The Streamlit dashboard has separate **Lymph-node spread** and **Recurrence**
tabs. A readiness panel distinguishes cohort preparation, verified data
definitions and the presence of a frozen model. Each task shows its actual
cohort audit. Trained experiments enable case-level measured inference,
model comparisons and CSV/report exports. No probabilities are simulated
when a model is unavailable.

Nodal pathology does not identify individual affected nodes. Clinical N stage
is absent, so this dataset cannot establish occult spread among patients
whose conventional staging is negative. Do not infer distant-organ spread
from these nodal labels.

Both tasks have published prior work. These implementations extend LungLab;
they do not establish a globally novel method. Joint subtype/nodal learning
remains a separate research hypothesis and is not implemented in this release.
Independent data and a focused method comparison are needed before stronger
research or clinical claims.
