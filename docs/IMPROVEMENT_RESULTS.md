# Attempts to exceed 83% accuracy

**Numeric target observed: 88.6% for the development-selected model; a later
bagged experimental candidate scored 91.4% on the same 35-patient holdout.**
This is an exploratory follow-up, not independently confirmed performance.
Accuracy, balanced accuracy and ROC-AUC are distinct metrics. None of the
models establishes validated superiority over the supplied paper's private
74-patient, 3–30 mm solid-nodule cohort.

The seven-model further refinement is documented in
[refinement results](REFINEMENT_RESULTS.md). Bagging gets 32/35 patients right,
one more than the default, but its development balanced accuracy is lower, so
the 88.6% model remains preferred under the frozen selection rules.

## Accuracy-focused follow-up

The earlier experiments selected by balanced accuracy. At the user's request,
the follow-up prioritizes overall accuracy. The finite 12-model protocol,
threshold grid and existing patient split were recorded before fitting in
`artifacts/radiogenomics_accuracy/protocol.json`. No patient was moved between
development and test, and test cases never fitted preprocessing, feature
selection, hyperparameters or thresholds. The 35-patient holdout had already
been inspected in the earlier experiment, so this is a follow-up benchmark.

Five outer patient folds evaluate each candidate on the 104 development
patients. Within each outer training fold, three inner folds jointly select
parameters and a probability threshold from 0.35/0.45/0.50/0.55/0.65/0.75/0.85.
The paper MLP retains its native decision. Accuracy selects the model family;
balanced accuracy and AUC break ties. All models are frozen before scoring.

| Model | Nested development accuracy | Holdout accuracy | Holdout balanced accuracy | Holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| Paper MLP, accuracy-selected | 66.3% | 71.4% | 50.0% | 0.673 |
| Original logistic | 77.9% | 80.0% | 55.4% | 0.628 |
| Rich logistic — development-selected | **80.8%** | **88.6%** | **76.8%** | **0.852** |
| Original SVM | 79.8% | 80.0% | 50.0% | 0.255 |
| Rich SVM | 77.9% | **88.6%** | 76.8% | 0.745 |
| Original LDA | 78.8% | 82.9% | 57.1% | 0.786 |
| Original Extra Trees | 79.8% | 82.9% | 67.9% | 0.755 |
| Original CatBoost | 77.9% | 80.0% | 50.0% | 0.719 |
| Clinical logistic | 79.8% | 80.0% | 50.0% | 0.648 |
| Clinical CatBoost | 78.8% | 80.0% | 50.0% | 0.699 |
| Original/clinical logistic fusion | 78.8% | 80.0% | 50.0% | 0.561 |
| Original/clinical CatBoost fusion | 79.8% | 82.9% | 57.1% | 0.719 |

**Rich logistic was selected from development results before this scoring.**
It uses median imputation, correlation pruning (0.95), a normal quantile
transform, ANOVA selection of 100 features and L2 logistic regression with
`C=0.1`. The frozen SCC threshold is **0.55**. Its 1,682 source inputs are
radiomics only; this winner does not require clinical data.

It correctly classifies **31/35 patients**: 27/28 AC and 4/7 SCC. Confusion matrix
`[[27,1],[3,4]]`; AC recall **96.4%**, SCC recall **57.1%**. Majority accuracy is
80%. Accuracy 95% bootstrap interval is **77.1%–97.1%**, so the result does not
establish that population accuracy exceeds 83%. Balanced accuracy is 76.8%,
AUC 0.852, F1 0.667 and Brier score 0.114.

The clinical join imports only age at diagnosis, sex, smoking status and pack
years. Source age timing requires confirmation before prospective CT use.
Histology supplies the target only; stage, molecular tests, treatment and
outcomes never become predictors.

## Expanded clinical-only follow-up

The official clinical release contains 68 additional eligible AC/SCC patients
without the downloaded radiomics. All were added to development: **172 training
patients (144 AC, 28 SCC)**, with the original **35 test patients** unchanged.
Five accuracy-focused clinical models were evaluated: logistic regression,
CatBoost, histogram boosting, Extra Trees and a logistic/CatBoost voting model.
The frozen development-selected voting model scored **80% holdout accuracy**,
with zero SCC recall. All five models scored 80% on the holdout.

The expanded development majority baseline is **83.7%**. Voting, Extra Trees
and histogram boosting matched it; this does not count as improvement despite
exceeding the numeric 83% threshold. Development target flags require accuracy
above both 83% and majority prediction. An early summary flag was corrected
to apply this requirement; predictions and model selection were unchanged.
Audit: `artifacts/radiogenomics_expanded_clinical` and
`data/radiogenomics_expanded_clinical.provenance.json`.

## Richer NSCLC Radiogenomics comparison

Joined independently published PyRadiomics features to official TCIA histology
by patient ID. Included 139 patients: 111 adenocarcinoma (AC), 28 squamous cell
carcinoma (SCC), with 1,682 original/filtered features. Clinical fields are
excluded. A seed-42 stratified split was locked before modelling: 104 development
patients and **35 holdout patients, 28 AC and 7 SCC**. The majority baseline is
**80% accuracy**.

Nested five-fold development CV with three inner folds selected models using
balanced accuracy. Preprocessing, feature selection and hyperparameter search
stay within training folds. All five final models were frozen before test scoring.

| Model | Development CV balanced accuracy | Holdout accuracy | Holdout balanced accuracy | Holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| Paper ReLU MLP architecture | 49.3% | 65.7% | 51.8% | 0.628 |
| Quantile-transformed RBF SVM | 52.2% | **77.1%** | **75.0%** | 0.837 |
| Shrinkage LDA | 58.7% | 68.6% | 69.6% | 0.842 |
| Robust regularized logistic regression | **70.6%** | 54.3% | 60.7% | 0.770 |
| Original/wavelet/filtered-view ensemble | 65.3% | 60.0% | 64.3% | 0.791 |

**Logistic regression was the development-selected winner.** The SVM's better
observed holdout score is exploratory; selecting it after inspecting these
results requires new independent test patients for a final claim. Its confusion
matrix is `[[22,6],[2,5]]` (rows/columns AC, SCC), with AC recall 78.6% and SCC
recall 71.4%. Its accuracy 95% bootstrap interval is **62.9%–91.4%**. Only seven
SCC test patients make estimates imprecise.

SVC precision/F1 were corrected from saved native class predictions because
its decision can differ from thresholding calibrated probabilities at 0.5.
The correction involved no retraining, threshold tuning or model reselection.

Audit: `artifacts/radiogenomics_rich/selection.json`, `external_report.json`,
locked splits, five fitted bundles, development/holdout predictions and plots.

## Other completed experiments

**Lung1:** seven model families were tested in nested patient CV on the 161
development patients; the previously inspected 41 test patients were excluded.
MLP, CatBoost, logistic regression, RBF SVM, shrinkage LDA, Extra Trees and
histogram gradient boosting all stayed below 83%. Highest CV accuracy was
70.2%, below the 74.5% majority baseline. Highest balanced accuracy was 55.3%.
Full results: `artifacts/improvement_development/search_report.json`.

**TabICLv2:** added a pretrained tabular foundation model, absent from the
paper, with a pinned/checksummed checkpoint, eight ensemble members and at most
50 training-selected features. Five-fold development CV excluded both existing
holdouts. Accuracy was 71.4% on Lung1, 76.9% on original Radiogenomics features,
and 77.9% on richer features. Corresponding balanced accuracies were 48.7%,
53.5% and 54.1%. It did not outperform majority prediction. The model authors
have not established its performance below 300 training samples.
Audit: `artifacts/foundation_development/development_report.json`.

**CT crop fusion:** downloaded and MD5-verified 62 TCGA CT patient arrays, each
containing five resized 224×224 slices. Excluded exact duplicate LUSC array 22,
identical to array 18. Included 61 unique stacks: 28 AC, 33 SCC. Clinical ID
order is unavailable; identifiers denote published array rows. Class comes
from homogeneous collection membership, with possible acquisition/source
confounding. The separate Radiogenomics crops were excluded because source
generation uses unordered set-derived patient order without a saved mapping;
we did not guess their histology labels.

Computed 93 crop intensity/texture features and 1,024 frozen ImageNet ResNet18
features from three overlapping three-slice views. Shape features are excluded
because HU and physical spacing are unavailable. Crop textures measure
bounding-box interiors rather than segmented tumour volumes. All five slices
remain in the same split. The MLP is an architectural adaptation, not an exact
reproduction of the paper's 3D radiomics.

Locked 45 development stacks and 16 holdout stacks (7 AC, 9 SCC). The adapted
MLP won nested development CV. It scored 56.3% holdout accuracy, versus 37.5%
for CNN/logistic regression and 43.8% for either hybrid logistic regression or
hybrid SVM. Majority accuracy is 56.3%. The fusion hypothesis failed.
Audit: `artifacts/tcga_crops_hybrid/external_report.json`.

## Evidence needed to confirm the improvement

1. Consistently extract features from raw HU CT with verified tumour masks,
   real clinical patient IDs, scanner settings and physical geometry.
2. Add more independent SCC cases and match the paper's solid-nodule inclusion
   criteria for a closer comparison. Keep class balance and split rules fixed.
3. Compare radiomics, a pretrained medical CT encoder and their fusion on the
   same patient split, with preprocessing fitted inside training folds.
4. Freeze a model/threshold, then evaluate it once on new external patients.
   Report accuracy, both subtype recalls, balanced accuracy, AUC, confidence
   intervals and the majority baseline.

The accuracy-focused radiomics result exceeds 83% numerically on our inspected
holdout. Fresh independent patients are needed to confirm it; changing the
split, reporting training accuracy or increasing majority-class prevalence
does not supply that evidence.

Sources: [supplied paper](https://doi.org/10.1038/s41598-024-83786-6),
[TCIA Radiogenomics](https://doi.org/10.7937/K9/TCIA.2017.7hs46erv),
[radiomics release](https://github.com/lawrence-chillrud/ece475),
[TabICL](https://github.com/soda-inria/tabicl),
[CT crops](https://doi.org/10.6084/m9.figshare.26037169),
[Verma et al.](https://doi.org/10.1016/j.crmeth.2024.100817).
