# TRIPOD+AI Reporting Checklist

**Study:** External validation and domain-shift analysis of maternal-risk prediction models in Nigerian clinical populations
**Type:** External validation of an existing prediction model (TRIPOD "validation" pathway)
**Reference:** Collins GS, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378

Page numbers to be completed against the typeset manuscript.

| # | Item | Status | Where addressed / note |
|---|---|---|---|
| **Title and abstract** ||||
| 1 | Title identifies study as developing/validating a prediction model, target population, and outcome | ☐ | Title must state "external validation" and "Nigerian primary care" explicitly |
| 2 | Structured abstract: objectives, data, methods, results, conclusions | ☑ | Draft complete |
| **Introduction** ||||
| 3a | Background and rationale, existing models referenced | ☑ | 6% sub-Saharan representation gap; MATRIACARE and prior benchmark models |
| 3b | Objectives, including whether development or validation | ☑ | Validation-only; stated in Objective |
| **Methods — Data** ||||
| 4a | Data sources, separately for development and validation | ☑ | Mendeley (n=1,072), Kaggle/UCI (n=452) for development; three Nigerian hospitals (n=382) for validation |
| 4b | Study dates, including follow-up | ⚠ | **Outstanding — collection dates for the three Nigerian sites are not yet documented** |
| 5a | Eligibility criteria for participants | ⚠ | **Outstanding — inclusion criteria used at each Nigerian site must be described** |
| 5b | Details of treatments received, if relevant | n/a | Triage-at-presentation; no treatment data |
| 5c | Study setting (care level, geography, number of centres) | ⚠ | Partially — three southwestern Nigerian hospitals (tertiary teaching, private group, specialist); needs care-level detail |
| **Methods — Outcome** ||||
| 6a | Outcome definition and how/when assessed | ⚠ | **Outstanding — how "high/mid/low risk" was assigned at each site (clinician judgement? protocol?) must be stated; the label definitions differ across sources** |
| 6b | Whether outcome assessors were blinded to predictors | ⚠ | **Outstanding — likely not blinded; must be disclosed as a limitation** |
| **Methods — Predictors** ||||
| 7a | Predictors defined, how and when measured | ☑ | Six shared: age, SBP, DBP, body temperature, heart rate, blood sugar; plus derived pulse pressure, MAP, shock index |
| 7b | Whether predictor assessors were blinded to outcome | ⚠ | **Outstanding — disclose** |
| 8 | Sample size and its justification | ☑ | Convenience sample; no a priori calculation. n=260 primary validation set; report as limitation with events-per-variable |
| 9 | Missing data handling | ☑ | Median imputation within fitted pipeline; implausible values set missing under pre-specified physiologic bounds; full counts in `audit_trail.csv` |
| **Methods — Analysis** ||||
| 10a | How predictors were handled in the analysis | ☑ | Continuous, untransformed; derived features specified in `experiment.py` |
| 10b | Model type, building procedure, internal validation | ☑ | Stacking ensemble (RF, gradient boosting, SVM) with logistic meta-learner; 5-fold stratified CV, seed 42 |
| 10c | How predictions were calculated | ☑ | `predict_proba`; high-risk vs not-high-risk binary endpoint |
| 10d | Performance measures | ☑ | AUROC, AUPRC, sensitivity at 80% specificity, Brier score, ECE; 2,000-sample bootstrap CIs |
| 10e | Model updating/recalibration, if any | ☑ | **None performed** — deliberate; recalibration is stated as future work |
| 11 | Risk groups, if created | n/a | None |
| 12 | Differences between development and validation data | ☑ | Core contribution: PSI, KS tests, domain-discriminator AUROC 0.991 |
| **Methods — AI-specific (TRIPOD+AI)** ||||
| 12a | Fairness: how subgroups were defined and assessed | ⚠ | **Partially — site-level breakdown done; age-band and BMI fairness analysis still to run** |
| 12b | Data preprocessing and feature engineering fully described | ☑ | `harmonize.py`, `experiment.py` |
| 12c | Software, packages, versions | ⚠ | **Outstanding — freeze `requirements.txt` with pinned versions** |
| 12d | Computational resources | ☑ | CPU only; runtime under 5 minutes |
| **Open science** ||||
| 13a | Funding and role of funders | ⚠ | To complete |
| 13b | Conflicts of interest | ⚠ | To complete |
| 13c | Protocol / registration | ⚠ | **Not registered — consider retrospective OSF registration; note the analysis protocol was pre-specified before results were seen** |
| 13d | Data availability | ⚠ | Public sources citable; **Nigerian data sharing requires institutional agreement — state the policy** |
| 13e | Code availability | ☑ | Repository with `harmonize.py`, `experiment.py`, `figures.py`, audit trail |
| **Patient and public involvement** ||||
| 14 | Whether patients/public were involved | ⚠ | **Outstanding — state plainly if none** |
| **Results** ||||
| 15a | Flow of participants, including exclusions | ☑ | `audit_trail.csv`; 1,921 duplicates and unlabelled/implausible records removed. **Render as a flow diagram** |
| 15b | Participant characteristics by dataset | ⚠ | **Outstanding — Table 1 of baseline characteristics per source** |
| 16 | Number of participants and outcome events | ☑ | Reported per arm |
| 17 | Model performance with confidence intervals | ☑ | Table of Arms A0/A1/A2/B/R |
| 18 | Model updating results | n/a | No updating performed |
| **Discussion** ||||
| 19 | Limitations | ⚠ | Must cover: small n; unverified label definitions across sites; FUTH prevalence and missing blood sugar; retrospective design; no outcome follow-up |
| 20a | Interpretation in context of objectives and evidence | ☑ | Calibration failure dominates discrimination loss |
| 20b | Generalisability | ☑ | Explicitly bounded to three southwestern Nigerian sites |
| 21 | Implications for practice and future research | ☑ | Local data collection over external corpus scale; recalibration before deployment |

**Legend:** ☑ addressed · ⚠ outstanding · n/a not applicable

## Critical path before submission

The ⚠ items on **6a (outcome definition), 5a (eligibility), and 4b (dates)** are the ones a methods reviewer will reject on. Blood pressure and heart rate are objective, but "high risk" is a *judgement* — and if the three Nigerian sites assigned it by different criteria, the reverse-transfer and site-level results carry a confound the current design cannot separate. Get the labelling protocol from each site in writing before the paper goes out.
