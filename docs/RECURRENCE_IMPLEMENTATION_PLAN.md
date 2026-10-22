# Recurrence prediction implementation plan

Prepared 5 October 2026. Scope: add a separate research module to LungLab that estimates the cumulative probability of a first recorded recurrence within two years after initial surgery, using preoperative tumour radiomics and a small set of baseline inputs. Reuse the imaging and provenance infrastructure; retain the existing subtype model and its results.

**Implementation status (5 October 2026):** cohort preparation, per-patient
date audit, competing-risk model code, protocol config, CLI and a dashboard tab
are implemented. The source audit writes 115 provisional cohort rows, but
training is blocked because the source does not verify the CT-to-surgery/radiomic
scan match or recurrence-free assessment through last-known-alive. No model has
been fit and no recurrence performance is claimed. The implementation uses a
small SciPy ridge Cox pair in place of an additional survival package; the
outcome estimand and evaluation protocol remain unchanged.

The preliminary counts below are superseded by the generated audit in
`artifacts/recurrence_audit/audit.json`. Neither set of counts is a validated
outcome label or performance estimate.

**Clinical question and first release**

The proposed prediction point is initial surgical management. The primary horizon is 730 days after surgery; 365 days is a secondary horizon when follow-up supports it. The target is first local, regional, or distant recurrence before death. Death before a recorded recurrence is a competing event. Predictions describe outcomes under the care represented in the study, and cannot estimate the benefit of choosing a different treatment.

The first cohort will use the existing Radiogenomics patients with recorded M0 disease, surgery information, an eligible preoperative CT, and interpretable follow-up. The primary imaging window is 1–90 days before surgery. A 1–180-day sensitivity analysis is prespecified and will not replace the primary analysis based on its score. Pathological M stage is a retrospective eligibility field; it is excluded from model inputs. Do not describe all cases as confirmed curative R0 resections: margin status is unavailable.

Clinical deployment and treatment recommendations are outside this release. Demonstrating added predictive value and validating the module on independent patients are separate milestones.

**What the preliminary audit established**

Sources are `data/radiogenomics_subtype.csv`, its preserved split manifest, and `data/raw/nsclc_radiogenomics/clinical.csv`.

| Finding | Current count or range | Consequence |
| --- | --- | --- |
| Patients with existing radiomics | 139 | Starting cohort before exclusions |
| Recorded recurrences | 37 | Keep model complexity small |
| Deaths without recorded recurrence | 26 | Use competing-risk methods |
| Recurrence dates | All 37 parse | Date meaning still needs verification |
| CT and last-known-alive dates | All 139 parse | Last-known-alive may differ from last recurrence assessment |
| Death dates | All 48 parse | Validate status/date consistency |
| Recorded CT-to-surgery interval | 1–735 days; median 44 | Audit long intervals and match features to the correct CT |
| Recorded M1 disease | 4 patients | Exclude from the primary nonmetastatic cohort |
| Current audited M0 cohort with provisional CT within 90 days | 115 patients; 26 recurrences and 24 competing deaths | Counts depend on unverified surgery-date interpretation and recurrence assessment |
| Current 180-day sensitivity cohort | 120 patients; 31 recurrences and 24 competing deaths | Sensitivity only; cannot replace primary cohort based on scores |
| Current preserved development partition | 84 patients; 18 recurrences and 17 competing deaths | Three-fold nested CV is feasible only after the semantic gate passes |
| Current preserved test partition | 31 patients; 8 recurrences and 7 competing deaths | Exploratory evaluation with substantial uncertainty |

If surgery is provisionally computed as CT date plus the recorded interval, one patient has a death date before surgery and one has last-known-alive before surgery. Flag these records for reconciliation. Do not silently change signs, clip negative durations, or substitute CT date for surgery date.

The generated audit also excludes one recurrence-positive patient whose recurrence date is after the recorded last-known-alive date. Its 1–180-day sensitivity cohort has 120 patients and 31 recorded recurrences. The 730-day observation counts are 55 development and 18 test patients; these counts describe recorded time, not verified recurrence surveillance.

The source publication confirms that dates are shifted for anonymization and retain within-patient chronology. Do not compare shifted calendar dates across patients to construct a temporal split. [Dataset methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6190740/)

**Phase 1 — establish the outcome contract and audit the cohort**

Create `lungml/recurrence_data.py` with an outcome loader independent of the binary `Dataset.y` contract in `lungml/data.py`.

1. Confirm that the published radiomic row corresponds to the CT recorded in the clinical table. Record the relevant source evidence.
2. Verify the direction and interpretation of `Days between CT and surgery`. Derive an index date only when this interpretation is supported; keep its derivation and confidence explicit.
3. Determine whether last-known-alive also represents the last recurrence assessment. A survival follow-up date alone does not establish recurrence ascertainment. If this cannot be supported, preserve the audit and withhold the postoperative recurrence model until suitable follow-up information is available.
4. Audit surgery, recurrence, death, and follow-up chronology; baseline metastatic disease; treatment-naive imaging; conflicting records; duplicate patients; and feature availability. Record an exclusion reason for each excluded row. Do not change the inclusion rules after inspecting model results.
5. Preserve the existing development/test assignment for retained patients. Never move excluded test patients or replacement patients into development.

The outcome table contains `patient_id`, `sample_id`, `index_date`, `ct_date`, `time_days`, `event_type`, `eligible`, `exclusion_reasons`, and `outcome_source`. Keep raw dates and derivation flags in the audit table, outside the feature matrix. Encode events as 0 = right-censored, 1 = first recurrence, 2 = death before recurrence. The current audit provisionally censors at last-known-alive but disables fitting until recurrence-free follow-up is verified; it treats documented recurrence/death on the same day as recurrence first.

Deliver `data/recurrence_outcomes.csv`, `data/recurrence_features.csv`, a provenance sidecar, and `artifacts/recurrence_audit/audit.json`. Raw sources remain unchanged. The audit reports exclusions, first-event counts, censoring, follow-up support at both horizons, and unresolved records. Feature and outcome joins must be one-to-one with patient IDs preserved as strings.

Completion gate: the index date and recurrence ascertainment have documented interpretations; retained records have valid positive durations and consistent event types. Missing semantic evidence is a data limitation, not permission to invent labels. The binary “Recurrence” field by itself is insufficient for two-year prediction.

**Phase 2 — freeze the protocol and implement modest baseline models**

`lungml/configs/recurrence.yaml` and `lungml/recurrence_models.py` implement the frozen protocol. The configuration is included in installed packages. The recurrence dependency group records SciPy, which supplies the bounded-memory L-BFGS optimizer used by the small ridge cause-specific Cox model. This avoids changing the existing subtype environment for an additional survival package. Breslow ties estimate each cause-specific baseline hazard; an Aalen–Johansen estimator supplies the nonparametric competing-risk reference.

Prespecify three comparisons:

| Model | Inputs | Role |
| --- | --- | --- |
| Aalen–Johansen estimate | Training first-event outcomes | Cohort-level reference without individual predictors |
| Regularized cause-specific Cox pair | Age at diagnosis and log tumour volume | Simple baseline; age must be available at the prediction point |
| Radiomics-augmented Cox pair | Same baseline plus 1 or 2 principal components from original non-shape radiomics | Test incremental imaging value with at most four predictors per cause |

Fit separate regularized hazards for recurrence and death before recurrence, then combine both into cumulative incidence curves. Do not label one minus recurrence-only survival as recurrence probability. Verify that both predicted incidences and event-free survival form a valid probability distribution.

Use median imputation, scaling, correlation pruning, and PCA fitted only on the current training fold. Keep tumour volume outside the PCA block. Start with original features to limit scope; rich filtered features are a later experiment. The available volume feature is `original_shape_MeshVolume`; its source unit and exact CT match remain unverified and are recorded as limitations in the model input schema. Unknown histology, postoperative pathological stage, recurrence information, survival dates, treatment received after the prediction point, and molecular test results cannot enter the feature matrix. Reuse radiomic allowlists, not the current subtype target or its feature-ranking algorithm.

Use ridge penalties `alpha = [0.1, 1.0, 10.0]`, a shared penalty choice across the two causes, and PCA components `[1, 2]`. Prefer fewer components and stronger regularization when scores tie. Record convergence failures. SMOTE, binary accuracy optimization, random forests, deep models, and broad model-family searches are outside the initial protocol.

**Phase 3 — train and evaluate with censoring and competing events**

Create `lungml/recurrence_train.py` and `lungml/recurrence_evaluation.py`.

Use the preserved development partition for three outer folds and three inner folds, stratified by first-event type, with seed 42. Every training fold must contain at least five recurrence events and five competing deaths. If exclusions make that impossible, stop and report insufficient event support rather than modifying the protocol using scores. Save the finite protocol before the first fit.

Select hyperparameters using the lowest inner-CV inverse-probability-of-censoring-weighted Brier score for recurrence cumulative incidence at 730 days. Compare the two individual models using nested development predictions under that same metric; retain the simple baseline when radiomics does not improve it. Keep the cohort reference alongside both models. Selection scores across these comparisons remain development estimates.

Implement evaluation for the recurrence cumulative-incidence target:

- Primary: IPCW Brier score at 730 days, with a paired difference against the simple baseline.
- Secondary: competing-risk time-dependent AUC at supported horizons, calibration against Aalen–Johansen estimates, and 365-day Brier score.
- Supplementary: cause-specific concordance, event counts, risk distributions, and uncertainty in model comparisons.

For horizon-specific metrics, cases have recurrence first by the horizon. Controls include patients with a competing death first by the horizon and patients observed recurrence-free through it. Patients censored earlier have an unobserved horizon outcome. Estimate censoring weights from training data only. Existing binary survival AUC/Brier utilities must not be reused without verifying these competing-risk definitions.

Require positive training censoring support, estimated censoring survival of at least 0.10, and at least ten training patients still at risk at a requested horizon. Mark unsupported horizons unavailable; do not extrapolate or choose a more favorable horizon after scoring. Inspect subgroup results descriptively without tuning separate thresholds.

Freeze models and the selected model using development evidence before scoring the retained legacy test patients once. Label that evaluation exploratory because these patients and their imaging were used in earlier project development. Save all candidates and failures. Bootstrap test patients for conditional metric intervals; those intervals do not include model-selection uncertainty. Individual prediction intervals require a separate development-bootstrap refitting procedure and must not be fabricated from between-model disagreement.

**Phase 4 — expose inference and reports**

`lungml/recurrence_predict.py` and `lung-ml recurrence prepare`, `train`, and `predict` implement recurrence-specific bundles and prediction output rather than forcing recurrence into subtype probabilities or labels.

Each bundle records task/schema version, endpoint and index definition, input names and unit caveats, preprocessing, both cause models, supported horizons, training IDs, source/data/split hashes, parameters, dependency versions, and source-code hashes. The CLI writes frozen `protocol.json`, `selection.json`, nested development predictions, models and—only after an explicit evaluation request—test predictions and an exploratory report. Output directories are immutable after selection/evaluation.

Output patient ID, recurrence cumulative incidence at supported horizons, the competing-death incidence, event-free survival, model version, and input quality/support flags. Inference accepts features and available baseline covariates; it never requires outcome dates. Validate IDs, units, missingness, extraction compatibility, and cohort applicability. An arbitrary Cox ranking score is not a percentage probability.

Use explicit ID joins when saving predictions. Validate raw CSV headers before pandas renames duplicates. Add regression coverage for numeric IDs and shuffled rows, addressing the input/export problems found during the project review in the new workflow.

**Phase 5 — add a recurrence view to the dashboard**

Extend `app.py` and the report adapter to show recurrence runs with their own schema. Present the prediction horizon and surgery-based time origin, model comparisons, follow-up support, and cumulative incidence curves. Display subtype and recurrence as separate outputs.

Show the simple baseline alongside the radiomics candidate so the added contribution is visible. If prediction intervals are unavailable, say so. Suppress unsupported horizon estimates. Keep cohort-level confidence bands distinct from individual uncertainty. Feature associations can explain model behavior but do not establish why a particular patient will recur. Avoid categorical high/low risk thresholds or follow-up schedules until their intended action and validation are defined.

Provide held-out example inference and downloadable prediction/report files. The research context and limitations should be concise; implementation details belong in the methods view.

**Phase 6 — verify, document, and assess the result**

Add focused tests in `tests/test_recurrence_data.py`, `tests/test_recurrence_models.py`, `tests/test_recurrence_evaluation.py`, and `tests/test_recurrence_pipeline.py` before treating the module as validated.

- Outcome tests cover recurrence first, death first, early censoring, missing assessment, contradictory dates, ties, and invalid durations.
- Leakage tests cover patient boundaries, fold-fitted preprocessing/PCA and censoring weights, predictor allowlists, and exclusion rules.
- Numerical tests compare cumulative incidence against analytic synthetic examples or an independent trusted reference, including competing deaths and tied events. Curves must be monotone, bounded, and sum with survival to one.
- Metric tests verify the treatment of competing deaths and censored patients at the horizon, training-only weights, and unsupported horizons.
- Integration tests verify serialization, shuffled/numeric IDs, prediction/report agreement, immutable evaluations, and recurrence dashboard rendering. Recheck existing subtype tests and dashboard behavior.

Update README, DATASETS, ROADMAP, and add `docs/RECURRENCE_RESULTS.md` after an experiment actually completes. Publish unsuccessful results too. A functioning module and an improvement in predictive performance are different acceptance outcomes; a baseline win is a valid result.

The final research milestone is a fresh surgical cohort with harmonized CT extraction, matching eligibility, and verified recurrence surveillance. The existing Lung1 survival labels cannot validate postoperative recurrence: its population and endpoint differ. Plan an independent evaluation before claiming clinical usefulness or superiority.

**Suggested delivery order**

| Milestone | Reviewable deliverable | Dependency |
| --- | --- | --- |
| 1 | Outcome audit and final inclusion flow | Date and recurrence-assessment semantics |
| 2 | Frozen protocol, baseline models, validated cumulative incidence | Milestone 1 passes |
| 3 | Nested development report and frozen models | Numerical and leakage checks pass |
| 4 | One exploratory legacy-test report and working CLI inference | Selection frozen |
| 5 | Dashboard, documentation, and regression checks | Report and predictions agree |
| 6 | Independent evaluation | New suitable cohort acquired |

Begin with Milestone 1. Do not promise model accuracy or an implementation deadline until the outcome semantics and usable event count are resolved.
