# What to add next

**Software delivery, 10 October 2026:** the CT/mask workbench, physical geometry
inspection, ±1 mm segmentation perturbation, case explanations, training-only
reference ranges, printable reports, local HTTP API and frozen external-cohort
evaluation are now implemented. See [the runbook](RUNBOOK.md) and
[explicit completion status](PROJECT_STATUS.md). The remaining data and
validation requirements below still apply.

**10 October 2026:** lymph-node involvement and recurrence workflows are both
implemented and visible in the dashboard. Nodal preparation writes 120
provisional eligible rows, with 24 node-positive patients. Its fitting gate
requires verified CT/surgery timing and scan correspondence. The recurrence
gate additionally requires recurrence-assessment semantics. Neither extension
has a trained model for the current source. See
[clinical extension instructions](CLINICAL_EXTENSIONS.md). Joint subtype/nodal
learning remains a proposed experiment; this release implements separate
nodal baselines and does not claim method novelty.

Correlation filtering, regularized linear/SVM models, radiomic-view ensembles,
TabICLv2 and frozen CNN/crop-texture fusion have now been implemented and tested.
Those initial experiments did not achieve the 83% target. The subsequent
accuracy-focused radiomics model scored **88.6% on the preserved 35-patient
holdout**, with 80.8% nested development accuracy. This inspected holdout result
is exploratory. See [improvement results](IMPROVEMENT_RESULTS.md).
The next priority is confirmation on fresh independent patients, consistent
raw CT preparation and additional minority-class cases.

A further seven-model comparison tested calibration, bootstrap bagging,
elastic net, PCA and stacking. Bagged logistic regression reached **91.4%
(32/35)**, while tying development accuracy and lowering development balanced
accuracy relative to the default. The default remains 88.6%; SCC test recall
remains 4/7 for both models. See [refinement results](REFINEMENT_RESULTS.md).

A separate competing-risk recurrence extension has been implemented through
cohort audit, a locked cause-specific Cox protocol, CLI inference and a
dashboard view. Its audit currently finds 115 provisional eligible patients,
but no recurrence model has been fit: the CT-to-surgery/scan match and
recurrence-free follow-up date are not verified by the source. Verify those
definitions or obtain recurrence-assessment dates before opening the training
gate. See [the recurrence plan](RECURRENCE_IMPLEMENTATION_PLAN.md).

## Highest-value project additions

| Addition | Why it helps | Data/work needed | How to evaluate |
| --- | --- | --- | --- |
| Post-surgical recurrence prediction | Estimates recurrence probability over time and treats death as a competing event | Recurrence-free surveillance dates and a verified surgery/CT match; current source definitions are incomplete | IPCW Brier score, competing-risk AUC, calibration, preserved split, then independent cohort validation |
| Two-stage system: malignancy then subtype | Connects the subtype paper to the requested cancer detection task | Benign/malignant diagnoses; train a separate malignancy model before subtype | Sensitivity, specificity and end-to-end errors; test both stages together |
| 3D CT viewer with tumour overlay | Makes input and mask quality inspectable during a demo | Local CT/mask upload; HU windowing and slice navigation | Correct orientation, spacing and overlay alignment |
| Independent external validation | Tests whether the system generalizes beyond Lung1 | NSCLC Radiogenomics or another hospital cohort with identical extraction | Patient AUC, per-class recall, calibration, CIs and cohort drift |
| Segmentation stability | Checks whether slight mask changes change the prediction | Multiple annotators or controlled erosion/dilation; repeatable extraction | Feature ICC and prediction stability; compare tumour volumes |
| Local feature explanations | Explains which measurements influenced a case | CatBoost SHAP with original feature names and units | Attribution checks and stability; do not present causal explanations |
| Quality and uncertainty review | Flags bad masks, missing measurements and uncertain cases | Geometry checks already implemented; add cohort-range and acquisition checks | Flag rates and failure capture; review coverage versus error |
| Exportable case report | Makes the project easy to present and audit | Probability, subtype, input quality, model/data version and overlay | Reproduce outputs from a saved case; show research context |

The paper already uses SHAP and decision curves. Adding those is valuable
implementation work but should not be described as a method absent from it.
CatBoost, leakage-controlled experiments and a usable inspection dashboard
are additional project contributions; none alone proves clinical utility.

## Stronger modelling experiments after data quality is established

1. **Feature stability and redundancy filtering.** Remove unstable features
   using training-only repeat segmentation data, then prune strongly correlated
   features inside CV. Compare the same CatBoost with and without filtering.
   Do not filter using test labels or full-cohort statistics.
2. **Radiomics plus deep CT embeddings.** Use a pretrained 3D medical-imaging
   encoder on tumour-centred CT patches, then combine its embeddings with
   radiomics in a regularized classifier. This needs reliable raw CT volumes,
   documented preprocessing and enough independent patients. Compare radiomics,
   embeddings and fusion on the same patient split.
3. **Clinical fusion.** Add age, smoking history and pre-diagnosis clinical
   variables in a separate model. Histology and post-diagnosis fields must never
   become inputs. Report radiomics-only, clinical-only and combined performance
   so any added value is clear.
4. **Scanner robustness.** Assess performance by scanner, reconstruction kernel,
   lesion size and site. Fit harmonization only on development data with an
   explicit policy for unseen sites; do not use test cohort statistics casually.
5. **Conformal prediction.** Reserve independent calibration patients to output
   a set of possible subtypes and measure coverage/set size. Prediction-set
   coverage depends on exchangeability and is not a guarantee under site drift.

A large CNN trained from scratch on 202 patients is a poor first experiment.
Data consistency, class balance and an independent test set remain priorities.
The selected accuracy-focused model has 96.4% AC recall but only 57.1% SCC
recall on seven SCC test patients. Minority-class recall and external stability
need improvement alongside confirmation of the overall accuracy.

## Localization and segmentation extensions

A nodule detector would let users supply a full CT rather than an existing
tumour mask. Train/evaluate it separately on nodule annotations, with free-
response ROC and false positives per scan. A 3D U-Net or nnU-Net could automate
segmentation when suitable masks and compute are available. Evaluate Dice and
boundary error as well as downstream classifier performance.

Do not infer a cancer subtype for every automatically detected nodule. First
evaluate its malignancy and segmentation quality; propagate uncertainty and
errors through the complete system. A detector finding a lesion is not itself
a cancer diagnosis.

## Practical delivery order

1. Audit and consistently re-extract raw Lung1 CT features; validate on an
   untouched external cohort. Include a majority-class and logistic-regression
   benchmark so weak models are visible.
2. Use the implemented CT viewer, mask quality checks, case explanations and
   exportable research report to inspect a consistently extracted new cohort.
3. Obtain confirmed benign/malignant examples and build/evaluate the first
   classification stage.
4. Add detector/segmentation automation and CT embedding fusion only when the
   necessary annotations and compute are available.

For the college project, the deliverable now includes a paper-derived baseline,
a development-selected alternative with observed 88.6% accuracy on our inspected
holdout, an experimental 91.4% bagged candidate, transparent unsuccessful
experiments and an extension plan. Report
that result as exploratory on public data, with its confidence interval and
both subtype recalls; independent confirmation remains outstanding.
