# Project delivery status — 12 October 2026

The project runs as a local subtype research application.
It is **not 100% complete as a validated three-endpoint prediction system**.
The recorded-recurrence classifier is fitted and available for live inference.
Nodal and fixed-horizon recurrence training still require verified source definitions.
Software cannot create that evidence.

The final acceptance run passed **103 tests in 24.06 seconds** on 10 October 2026.
The recurrence-classifier update passed **107 tests in 14.51 seconds** on
12 October 2026, including frozen model checks, API inference, exact explanations,
and proof that changing test inputs/labels cannot change development selection.
The running HTTP service matched local inference for an existing public-cohort case.
An installed wheel passed CT extraction and API health checks outside the project folder.
These checks establish software behavior. They do not validate clinical performance.

| Deliverable | Current state | Evidence |
| --- | --- | --- |
| Paper-derived MLP and CatBoost baseline | Implemented and fitted | Saved Lung1 models, patient splits and reports |
| Accuracy-focused subtype models | Implemented and fitted | Default observed 31/35; experimental bagged candidate 32/35 |
| Leakage-controlled selection | Implemented | Nested development CV, training-only transforms, frozen thresholds |
| Real-time research dashboard | Implemented | Actual completions/timings, retained case inspection, CSV export |
| Nodal extension software | Implemented; current-cohort fitting blocked | Audited 120 provisional patients, nested baselines, frozen scoring and task UI |
| Recorded-recurrence prediction | Trained and running; exploratory | 139 labels, 104 development/35 test; clinical accuracy 48.6%, balanced accuracy 58.0%, AUC 0.592; 8/10 recurrences found, 16 false positives |
| Time-based recurrence extension | Implemented; current-cohort fitting blocked | Audited 115 provisional patients, competing-risk models, frozen scoring and task UI |
| CT viewer and tumour overlay | Implemented | Three planes, windowing, aligned overlay and physical 3D boundary view |
| CT/mask quality inspection | Implemented | Geometry rejection, volume, connected components and boundary review |
| Segmentation perturbation | Implemented | Actual original/±1 mm feature extraction and export |
| Case explanations | Implemented | Exact logistic-score reconstruction; labelled sensitivity fallback |
| Cohort-range review | Implemented | Model-bound profiles from fitted training patients only |
| Printable case reports | Implemented | Self-contained HTML, prediction/input hashes, escaped text and JSON payload |
| Local HTTP API | Implemented | Registered models, schemas, readiness, prediction, explanation and report routes |
| Independent classification evaluation software | Implemented; fresh cohort needed | Frozen model/threshold, overlap rejection, metrics and bootstrap intervals |
| Team guide and operating instructions | Implemented | Interactive offline guide, CLI help and runbook |

## Work that remains

1. **Verify CT identity and timing.** Establish that each radiomic scan matches
   its clinical CT date. Verify the CT-to-surgery interval direction.
   This blocks fitting the current nodal and time-based recurrence cohorts.
2. **Obtain recurrence-assessment dates or their documented equivalent.**
   Last-known-alive dates do not establish recurrence-free surveillance.
   This blocks time-based recurrence fitting. The trained recorded-label
   classifier has no fixed follow-up horizon and does not resolve this requirement.
3. **Train and freeze both extensions after those checks pass.**
   The implemented commands will then fit baselines and evaluate their preserved splits.
   Their predictions and measured performance must come from those actual runs.
4. **Validate on fresh, compatible patients.** All existing subtype test results
   are exploratory. These patients were inspected during earlier experiments.
5. **Establish clinical utility and method novelty separately.**
   A working implementation cannot establish either claim.
   Nodal prediction and competing-risk modelling have prior literature.

## Future research scope

Automatic tumour detection, automatic segmentation, trained malignancy detection,
joint subtype/nodal learning and matched 3D embedding fusion remain future work.
They require suitable annotations, compatible raw CT and separate evaluation protocols.
They are not hidden features of this release.

Use [the runbook](RUNBOOK.md) to operate the completed software.
Use [extension instructions](CLINICAL_EXTENSIONS.md) to resolve the data gates.
