# LungLab: lung CT radiomics classification

A runnable ML project based on the supplied Selvam et al. paper, with its best
ReLU MLP architecture, accuracy-focused radiomics models, additional clinical
and fusion experiments, patient-level evaluation, CT/mask feature extraction,
and a Streamlit dashboard.

The research application also includes a CT/mask workbench, segmentation
perturbation, case explanations, training-range review, printable case reports,
a local HTTP API and frozen external-cohort evaluation. See
[operating instructions](docs/RUNBOOK.md) and [delivery status](docs/PROJECT_STATUS.md).
A trained exploratory classifier also predicts **source-recorded recurrence**
from postoperative clinical information, with an optional CT-radiomics model.
Fixed-horizon recurrence and nodal fitting still require verified source definitions.

**Highest observed experimental accuracy is 91.4% (32/35 patients)** with a
12-estimator bagged logistic model. The development-selected regularized
logistic model remains the default at **88.6% (31/35)**: bagging tied development
accuracy but had lower development balanced accuracy. Both exceed the numeric
83% target and the 80% majority baseline on this test set. These follow-up
results are **exploratory**: these test patients were inspected
in earlier experiments, and the source paper used a different cohort. It does
not establish validated superiority over the paper. Full results are in
[improvement results](docs/IMPROVEMENT_RESULTS.md) and the
[seven-model refinement](docs/REFINEMENT_RESULTS.md).

The implemented and trained task is **adenocarcinoma versus squamous cell
carcinoma**, assuming a known segmented tumour. A benign-versus-malignant task
is supported by the training code but requires an appropriate separate dataset.
Neither model detects or localizes an unknown tumour in a full CT scan.

## Lymph-node involvement and recurrence

Both clinical extension workflows are implemented, with separate targets,
audits, model protocols, inference and dashboard views. **The recorded-recurrence
classifier is trained and running.** The nodal and fixed-horizon competing-risk
models still await verified source definitions. See
[clinical extension instructions](docs/CLINICAL_EXTENSIONS.md).

The new nodal workflow estimates pathological **N0 versus N1/N2/N3** from
matching preoperative tumour features. Prepare its cohort with:

```bash
.venv/bin/lung-ml nodal prepare
```

The audit finds 139 source patients (113 node-negative and 26 node-positive).
Within the provisional 1–90 day preoperative window, 120 remain eligible,
including 24 node-positive patients. Training requires documented scan/timing
verification. After verification, `nodal train` compares a prevalence
reference, a volume benchmark and radiomics logistic regression using nested
development CV. `nodal evaluate` scores the frozen experiment once;
`nodal predict --model ... --data ...` exports probabilities. The packaged
protocol is `lungml/configs/nodal.yaml`.

Nodal involvement is a separate endpoint from later recurrence. Nodal labels
do not identify individual affected nodes, establish occult spread, or
describe distant-organ metastases. These are research extensions, not newly
established methods or treatment recommendations.

## Recurrence prediction extension

A working **recorded-recurrence classifier** is available in the dashboard's
**Recurrence** tab. Select an example patient, enter a postoperative profile,
or upload a feature CSV, then click **Predict recurrence**. It returns actual
probabilities, feature explanations, input checks, and CSV/HTML reports.

Two L2 logistic models were trained on 104 development patients and scored on
35 preserved test patients. The clinical model uses age, sex, smoking,
pathological T/N categories, grade, lymphovascular invasion and pleural invasion.
The second adds original CT radiomics, selecting 10 inputs inside training folds.
Model selection uses five-fold development Brier score; thresholds maximize
development out-of-fold balanced accuracy on a fixed 0.10–0.60 grid.

The development-selected clinical model has **48.6% test accuracy**, **58.0%
balanced accuracy**, and **AUC 0.592**. It finds **8/10 recorded recurrences**
with **16 false positives**. The CT candidate has 65.7% accuracy, 61.0%
balanced accuracy, and AUC 0.636. The always-no reference has 71.4% accuracy
and 0% recurrence recall. These results are exploratory and separate from
the subtype model's 88.6% accuracy.

The target is the explicit yes/no recurrence field across varying follow-up.
**There is no one-year or two-year risk claim.** A source "no" does not
establish future freedom from recurrence. Surgical pathology is required.
Initial age/smoking-only test results were also inspected. Development
threshold metrics are optimistic because the threshold uses those same scores.

```bash
.venv/bin/lung-ml recurrence train-recorded --output artifacts/my_recorded_recurrence
.venv/bin/lung-ml recurrence predict-recorded --data my_patient_features.csv --output my_recurrence_predictions.csv
```

Frozen models, training-only input profiles, cohort, development/test
predictions and the measured report are in `artifacts/recurrence_recorded/`.
Existing frozen output directories cannot be overwritten.

A separate small-cohort training experiment deliberately fits an unrestricted
decision tree to **40 development patients**. It achieves **100% training
accuracy** and **68.6% accuracy on the separate 35 test patients**. Both appear
together under **Training fit on a smaller cohort** in the Recurrence tab.
The training score measures memorization and must not be presented as accuracy
on new patients. This experiment does not replace the development-selected model.
Its artifacts are in `artifacts/recurrence_training_fit/`. Reproduce it in a
new output directory with:

```bash
.venv/bin/lung-ml recurrence training-fit --patients 40 --output artifacts/my_training_fit
```

The separate **time-based** post-surgical workflow retains its audit:

A separate post-surgical recurrence workflow now builds a patient-level audit,
preserves the existing development/test split, and defines recurrence with
death before recurrence as a competing event. Start with:

```bash
.venv/bin/lung-ml recurrence prepare
```

This writes the cohort, feature/outcome tables and a detailed audit under
`artifacts/recurrence_audit/`. The current audit reports 115 provisional
eligible patients, but training is deliberately locked: the source does not
verify that its CT-to-surgery interval matches the radiomic scan or that a
recurrence-negative patient was assessed through the last-known-alive date.
The audit is not a validated recurrence model, and no outcome performance is
claimed. After those source definitions are independently documented, the
audit accepts the source references with `--surgery-date-evidence` and
`--recurrence-followup-evidence`; only then can the locked experiment be
started with `.venv/bin/lung-ml recurrence train`;
inference uses `.venv/bin/lung-ml recurrence predict --model ... --data ...`.
See [the implementation plan](docs/RECURRENCE_IMPLEMENTATION_PLAN.md) for the
model protocol, and [dataset notes](docs/DATASETS.md) for its current limits.
The packaged protocol is in `lungml/configs/recurrence.yaml`. Its censoring
support thresholds apply to training, scoring and bootstrap intervals; test
evaluation uses the thresholds saved with the frozen experiment and verifies
every saved model's hash before scoring.

## Run the completed project

For a new GitHub checkout, create the environment before starting the dashboard:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]' setuptools wheel
.venv/bin/streamlit run app.py
```

The repository includes the source code, prepared feature tables, saved models,
experiment reports, tests, and team guide. Raw source downloads, training caches,
temporary files, and the local Python environment are excluded. See the download
commands below to restore raw inputs for rebuilding experiments. CT feature
extraction requires the additional imaging dependencies described under
**Recreate the environment**.

For an interactive introduction for engineers without ML experience, open
[the team field guide](docs/team-guide.html) in a browser. This self-contained
HTML page works offline. It explains the pipeline, evaluation, result limits,
nodal and recurrence audits, and engineering responsibilities with diagrams
and teaching controls. Its base snapshot is dated 10 October 2026, with the
trained recorded-recurrence update added on 12 October 2026.

The local `.venv` is installed and the original and additional models are trained.

```bash
.venv/bin/streamlit run app.py
```

The LungLab dashboard runs saved models locally with a responsive navy-and-teal
interface. Choose an **Experiment** and **Model** in the sidebar, then use:

- **Live inference:** queue preserved test patients or upload a feature CSV,
  then click **Run live inference**. Watch actual case completions, measured
  prediction timings, class probabilities and experimental review flags. Inspect
  completed cases, explore the interactive batch chart and download predictions.
  Changing the model or input clears stale results. The whole queue is validated
  before inference begins.
- **Model comparison:** switch between accuracy, balanced accuracy and AUC,
  inspect the selected model's confusion matrix, and download the saved report.
  Existing ROC/PR/calibration and feature-importance figures remain available.
- **Case review:** inspect selected-feature contributions, training-range flags
  and missing inputs. Download a self-contained report with model/input hashes.
- **CT & mask:** inspect local volumes in three planes, adjust CT windows,
  rotate the tumour boundary, extract features and test ±1 mm mask changes.
  The initial phantom is explicitly synthetic and produces no cancer prediction.
- **Lymph-node spread:** inspect recorded nodal labels, the provisional
  preoperative cohort and timing audit. Verified frozen experiments enable
  model comparison, measured inference and probability exports.
- **Recurrence:** run trained recorded-recurrence classifiers on example
  patients, a manual postoperative profile, or a feature CSV. Inspect
  probabilities, explanations and measured performance. The time-based
  cohort audit remains available in an expander.
- **Clinical extension readiness:** check cohort, source-definition and trained
  model status for both extensions before opening their results.
- **Methods & data** and **Development experiments:** read metric explanations,
  input requirements, provenance and the evidence used for model selection.

The default `radiogenomics_accuracy` experiment shows the development-selected
88.6% model. Select `radiogenomics_refinement` to compare the seven further
variants, then choose **Bagged logistic · experimental** to inspect the 91.4%
candidate. Select `radiogenomics_rich` for the earlier five-model comparison, or
`nsclc_lung1_converged` for the original benchmark. The earlier `nsclc_lung1`
experiment is retained as a development run with convergence warnings; use the
converged run for inference. Live example runs do not create new validation
results or change the saved performance statistics.

```bash
.venv/bin/lung-ml predict \
  --model artifacts/radiogenomics_accuracy/rich_accuracy_logistic.joblib \
  --data data/radiogenomics_subtype.csv \
  --output predictions.csv
```

Predictions over the full training CSV demonstrate inference; they do not
constitute new validation. The held-out predictions are saved separately.

## Data already downloaded

`data/nsclc_subtype.csv` contains **202 real patients**, with 107 original tumour
radiomic features: 51 adenocarcinoma and 151 squamous cell carcinoma cases.

Histology labels are downloaded directly from TCIA's official Lung1 clinical
release. CT-derived feature values come from `hhfchoi/Radiomics`, a third-party
Slicer feature extraction of the same collection. Only **primary GTV-1 tumour
masks** are included. Features from other tumour/lymph-node masks and aggregate
masks are excluded. Histology, age, stage, survival, patient IDs, and other
clinical fields are excluded from model inputs.

The provenance JSON records URLs, the pinned repository commit, SHA-256 hashes,
inclusion counts, limitations and attribution. This download is a compact
**feature dataset**, not the complete 33 GB collection of raw DICOM images.
See [dataset guidance](docs/DATASETS.md).

Two additional datasets are prepared: `data/radiogenomics_subtype.csv` contains
139 patients (111 AC, 28 SCC) and 1,682 radiomic features;
`data/tcga_crops_features.csv` contains 61 unique published patient stacks
(28 AC, 33 SCC), 93 crop textures and 1,024 frozen ResNet18 features. One exact
duplicate CT array was excluded before splitting. The crop benchmark lacks
recoverable clinical patient-ID order and physical CT geometry; its provenance
is weaker than the radiomics cohorts. Neither dataset contains normal controls.

```bash
.venv/bin/lung-ml download
# To rebuild from the existing local downloads without network access:
.venv/bin/lung-ml download --prepare-only
```

## Models and results

The accuracy-focused run compares 12 model variants on the preserved
Radiogenomics split. Nested five-fold development CV with three inner folds
selects parameters and, for alternatives, a decision threshold to maximize
accuracy. Both subtype recalls, balanced accuracy and AUC remain reported.

The selected model uses median imputation, training-only correlation pruning
at 0.95, a normal quantile transform, 100 ANOVA-selected features, and L2
logistic regression (`C=0.1`). Its SCC probability threshold is **0.55**.
It uses radiomics only; age, sex and smoking inputs are unnecessary for this
winner. Development nested-CV accuracy is **80.8%**, not 88.6%.

| Model / baseline | Accuracy on the same 35 test patients | Balanced accuracy | AUC |
| --- | ---: | ---: | ---: |
| Paper-style MLP, accuracy-selected parameters | 71.4% | 50.0% | 0.673 |
| Selected richer-radiomics logistic regression | **88.6%** | **76.8%** | **0.852** |
| Richer-radiomics SVM | 88.6% | 76.8% | 0.745 |
| Follow-up bagged logistic — experimental candidate | **91.4%** | **78.6%** | 0.847 |
| Majority-class prediction | 80.0% | 50.0% | — |

The winning model classifies 27/28 AC cases and 4/7 SCC cases correctly. Its
accuracy 95% bootstrap interval is **77.1%–97.1%**, which includes 83%.
The model and threshold were frozen before this follow-up scoring; the holdout
was nevertheless inspected during previous experiments. New independent
patients are required to confirm performance.

The further bagged candidate combines 12 bootstrap-trained pipelines, each
selecting 40 features and fitting logistic regression with `C=0.3`. It averages
SCC probabilities and uses a development-chosen threshold of 0.55. It gets
28/28 AC and 4/7 SCC cases right: the extra correct prediction improves AC
recall, while SCC recall remains 57.1%. Its accuracy bootstrap interval is
**82.7%–100%**, which still includes 83%. Its nested development accuracy is
80.8%, with balanced accuracy 52.4% versus the default's 55.9%. Calibration,
elastic net, PCA and stacking did not improve on the default's test accuracy.
All seven results and the frozen selection are in
[refinement results](docs/REFINEMENT_RESULTS.md).

Clinical/fusion inputs are prepared in `data/radiogenomics_clinical_subtype.csv`.
A separate clinical-only experiment adds 68 eligible development patients
from the already downloaded official clinical release, retaining the same
35 test patients. It scored 80%, matching majority prediction. Its 83.7%
development majority baseline must not be presented as an improvement.

Original Lung1 benchmark:

- **Paper MLP:** median imputation → scaling → 10 LASSO-ranked features →
  training-only SMOTE → MLP with one 100-unit ReLU hidden layer and learning rate
  0.001. Unknown paper settings are documented in [paper review](docs/PAPER_REVIEW.md).
- **CatBoost candidate:** all 107 radiomic features, class weights, regularized
  shallow trees, development-only parameter search and grouped sigmoid
  calibration. Its classification threshold is chosen using out-of-fold
  development probabilities to maximize balanced accuracy.

On the original experiment's **41 held-out patients**:

| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Paper architecture | 0.756 | 0.568 | 0.590 | 0.211 |
| CatBoost candidate | 0.634 | 0.487 | 0.542 | 0.186 |

Lower Brier score is better. CatBoost improved development CV AUC and Brier
score, but **did not improve held-out discrimination or balanced accuracy**.
It is an implemented improvement hypothesis, not a demonstrated superior model.
Its CV-selected threshold also failed to improve balanced accuracy on this test
set. The paired bootstrap AUC difference includes zero. A majority-only SCC
classifier would score 75.6% accuracy here, so accuracy alone is misleading.

The paper's 83% accuracy and 88% AUC are reported on a different, smaller cohort.
They are not reproduced or directly comparable to these results. This public
cohort contains larger radiotherapy tumours and uses a third-party extraction
whose settings are not fully documented. The test set was inspected during a
convergence fix; the final run reuses that split and is a development benchmark.
An untouched external cohort is necessary for a final performance claim.

## Train a new experiment

Additional dataset/model commands are available:

```bash
.venv/bin/lung-ml download-rich
.venv/bin/lung-ml download-crops
.venv/bin/lung-ml experiment --cohort radiogenomics \
  --output artifacts/new_radiogenomics --evaluate
.venv/bin/lung-ml experiment --cohort crops \
  --output artifacts/new_crops --evaluate
.venv/bin/lung-ml accuracy \
  --output artifacts/new_accuracy_run --evaluate
.venv/bin/lung-ml accuracy-clinical \
  --output artifacts/new_clinical_run --evaluate
.venv/bin/lung-ml refine \
  --output artifacts/new_refinement_run --evaluate
```

Use a fresh output directory. The `experiment` comparisons use nested five-fold
development CV with three inner folds, select by balanced accuracy, freeze all
models, and evaluate a locked 25% holdout once. Dataset/split hashes and patient
overlap are checked. Completed holdouts cannot be rescored through this command.
Recreating the same split is reproducibility, not new independent validation.

The `accuracy` commands select by nested CV accuracy, with balanced accuracy
and AUC breaking ties. Alternative thresholds are searched over
0.35/0.45/0.50/0.55/0.65/0.75/0.85 using inner out-of-fold probabilities.
Their complete finite protocol is saved before fitting. The existing patient
split is preserved, and additional clinical patients are development-only.
The `refine` command applies the same selection rules to a baseline and six
further candidates. All results, including unsuccessful candidates, remain in
their reports. The highest test score is labelled separately and does not
override the development-selected model.

The seven-model development search and fixed TabICLv2 comparison are retained
in `artifacts/improvement_development` and `artifacts/foundation_development`.
They exclude previously inspected holdouts. Their entry points are
`lungml.improve.development_search` and `lungml.foundation_experiment.run`;
choose new output paths to rerun. The verified TabICLv2 checkpoint is stored in
`data/raw/tabicl`, with its pinned URL/hash recorded in the experiment report.

```bash
.venv/bin/lung-ml train \
  --data data/nsclc_subtype.csv \
  --output artifacts/my_experiment
```

The seed is 42. 20% of patients are reserved before preprocessing. The remaining
patients undergo 5-fold stratified patient CV. All measurements from a patient
stay together. Missing-value imputation, scaling, feature selection and SMOTE
are fitted only on each training fold. Model choice uses development CV alone;
test patients are never used to fit a saved model or choose a threshold.

Outputs include two model bundles, development CV tables, patient split IDs,
held-out predictions, bootstrap intervals, metrics, plots, feature importances,
dataset hash and dependency versions. Output directories must be empty to
prevent accidentally overwriting a completed experiment.

For multiple nodules with the same patient-level label, final scores average
their probabilities per patient. Conflicting labels within a patient are rejected.
Small-cohort confidence intervals will be wide. Selection CV scores are
optimistic because they selected the parameters; use held-out and external
metrics to estimate generalization.

## Train from your own CT scans

1. Export a 3D CT in Hounsfield units and an aligned tumour label mask using
   ITK-SNAP or 3D Slicer. Supported volumes include NIfTI and NRRD; a directory
   containing exactly one DICOM series can also be read as CT.
2. Copy `data/manifest.example.csv` to `data/manifest.csv` and fill in the real
   patient IDs, sample IDs, histology labels, image paths, mask paths and labels.
   Relative paths resolve from the manifest's directory.
3. Extract features and train models using a consistent extraction configuration:

```bash
.venv/bin/lung-ml extract \
  --manifest data/manifest.csv \
  --params configs/radiomics.yaml \
  --output data/my_ct_features.csv
.venv/bin/lung-ml train \
  --data data/my_ct_features.csv \
  --output artifacts/my_ct_experiment
```

`configs/radiomics.yaml` specifies project settings: original-image features,
25-HU bins, isotropic 1-mm resampling and no intensity normalization. They are
explicit assumptions, not recovered paper settings. CT and mask geometry must
match; empty, misaligned and non-3D masks are rejected. DICOM RTSTRUCT or SEG
must first be converted to a tumour label volume with 3D Slicer/dcmqi.

JPG/PNG screenshots cannot replace the paper's volumetric CT input. Likewise,
matching CSV column names do not make newly extracted CT features compatible
with the downloaded third-party features. Retrain on features produced with
the same settings used at inference.

For the stronger original/wavelet/LoG comparison on a **new** raw CT cohort:

```bash
.venv/bin/lung-ml extract --manifest data/manifest.csv \
  --params configs/radiomics-rich.yaml --output data/new_cohort.csv
.venv/bin/lung-ml experiment --cohort radiogenomics --data data/new_cohort.csv \
  --output artifacts/new_standardized_ct --evaluate
```

The first experiment reserves a patient split once and saves it beside the
features. It requires one primary tumour per patient. Do not mix native-spacing
downloaded features with this standardized extraction without validating their
compatibility.

For benign/malignant nodule classification, use labels `benign` and `malignant`
and pass `--task malignancy`. That task is not trained by the current dataset.

## Recreate the environment

Python 3.11 is recommended; this project supports Python 3.11 and 3.12.

```bash
uv venv --python 3.11
uv pip install -e '.[dev]' setuptools wheel
uv pip install SimpleITK pywavelets pykwalify six
uv pip install --no-build-isolation pyradiomics==3.0.1
uv pip install -e '.[advanced]'
```

PyRadiomics 3.0.1 omits NumPy from its isolated build requirements, so install
the core dependencies first and use the final command above for imaging.
`requirements.lock.txt` records the exact installed dependency versions.

```bash
.venv/bin/python -m pytest -q
```

Tests cover patient leakage, clinical-field exclusion, duplicate data, minority
SMOTE handling, both models' serialization/inference, metric definitions and
actual 3D radiomic extraction with geometry validation.

The suite also checks API inference, training-only reference profiles, exact
linear explanations, case-report escaping, physical viewer geometry and
external-cohort overlap rejection. Start the local API with
`.venv/bin/lung-ml serve`; its interactive documentation is at
`http://127.0.0.1:8000/docs`. Use `.venv/bin/lung-ml doctor` to inspect actual
installed dependencies and unresolved extension gates.

## Extensions

See [project roadmap](docs/ROADMAP.md) for additions, data requirements,
evaluation criteria and a practical order of implementation.
