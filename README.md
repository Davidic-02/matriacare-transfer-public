# When Benchmarks Do Not Travel

External validation and distributional-shift analysis of maternal risk
prediction models in Nigerian clinical populations.

A stacking ensemble trained on the public benchmarks that dominate maternal
risk modelling (Mendeley, Kaggle/UCI) is validated against clinical records
from three southwestern Nigerian hospitals, in both transfer directions.

## Headline results

| Arm | n | AUROC [95% CI] | ECE |
|---|---|---|---|
| External, internal 5-fold CV | 1,524 | 0.964 [0.955–0.972] | 0.045 |
| External → Nigerian (primary) | 260 | 0.865 [0.822–0.905] | 0.234 |
| Local ceiling, 5-fold CV | 260 | 0.944 [0.912–0.972] | 0.059 |
| Nigerian → external (reverse) | 1,524 | 0.843 [0.821–0.862] | 0.120 |

Discrimination degrades modestly under transfer; calibration degrades
five-fold. A model developed and evaluated within the Nigerian population scores higher than
models transferred into it, a gap not explained by training size
(`outputs/R3_size_matched.csv`); whether it reflects site-specific labelling is unresolved. In the highest blood-sugar tertile, discrimination is low (AUROC 0.581
[0.430–0.732]), which the pooled figure does not show; the subgroup's 76%
high-risk prevalence may itself depress AUROC.

The two public benchmarks are not interchangeable. Trained on Kaggle/UCI alone
the model transfers to the Nigerian records with no loss of discrimination
(0.920 internally, 0.920 externally); trained on Mendeley alone it loses 0.156
(0.972 to 0.816). Calibration error rises on Nigerian data for both. Kaggle/UCI
is not the distributionally closer source — the domain discriminator separates it
from the Nigerian records at 0.988 against Mendeley's 0.984, and Mendeley is the
closer source on outcome prevalence — so a reported transfer figure is a property
of the particular development corpus (`outputs/R9`–`R11`).

## Reproducing

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
PYTHONPATH=src python src/harmonize.py             # raw -> harmonised table + audit trail
PYTHONPATH=src python src/experiment.py            # transfer arms + shift diagnostics
PYTHONPATH=src python src/figures.py               # ROC, calibration, shift and site figures; PSI table
PYTHONPATH=src python src/table1_flow_fairness.py  # Table 1, flow diagram, fairness analysis
PYTHONPATH=src python src/revision.py              # peer-review revision analyses (R1-R6)
PYTHONPATH=src python src/revision_ci.py           # calibration intervals and seed stability (R7-R8)
PYTHONPATH=src python src/revision2.py             # between-source heterogeneity and per-source transfer (R9-R11)
PYTHONPATH=src python src/figure_overview.py       # methodological overview figure
```

Run from the repository root. All randomness is seeded (`SEED = 42` in
`src/common.py`). Runtime is a few minutes on CPU.

The arms involving Nigerian records cannot be reproduced without those
records; see Data availability below.

## Layout

| Path | Contents |
|---|---|
| `src/common.py` | Feature engineering, model definition, bootstrap metrics |
| `src/harmonize.py` | Five sources to one schema, with a counted audit trail |
| `src/experiment.py` | Transfer arms and distributional-shift diagnostics |
| `src/figures.py` | ROC, calibration, shift and site figures |
| `src/table1_flow_fairness.py` | Table 1, participant flow, subgroup fairness |
| `src/revision.py` | Matched-feature, size-matched, model-agnostic and glucose sensitivity analyses |
| `src/revision_ci.py` | Bootstrap intervals for calibration error and seed stability |
| `src/revision2.py` | Between-source heterogeneity and transfer from each external source alone |
| `src/figure_overview.py` | Methodological overview figure |
| `outputs/` | Figures, result tables, audit trail, TRIPOD+AI checklist |
| `data/` | How to obtain the public datasets. No patient data is distributed here. |

## Endpoint

High risk versus not-high risk — the one label definition all five sources
support. The Mendeley and FUTH sources carry no mid-risk category.

## Known data issues

- 1,921 of 3,474 records in the two public repositories are identical rows.
  Several variables take few distinct values, so some may be different patients
  rather than repeated entries. They are removed either way, because identical
  rows split across training and test partitions inflate performance. Counts in
  `outputs/audit_trail.csv`.
- The FUTH `Height` column mixes units (range 1.13–7.2) and is excluded from
  the shared schema.
- Glucose is truncated at a lower bound of 6.0 mmol/L in the Nigerian sources
  **and in Kaggle/UCI** (no values below 6.00). Mendeley is the only source with a
  lower tail, reaching 3.0 mmol/L, and it drives most of the glucose PSI of 2.55.
  This is a shared measurement convention, not a Nigeria-specific recording artefact.

## Data availability

This repository contains **no patient data**. The two public datasets must be
downloaded from their original repositories (see `data/README.md`). The
Nigerian clinical records are available from the corresponding author on
reasonable request, subject to permission from the contributing hospitals.

## Citation

Adekoya D., Akinbo R. S. "When Benchmarks Do Not Travel: External Validation
of Maternal Risk Prediction Models in Nigeria." Manuscript under review,
Journal of Future Artificial Intelligence and Technologies.

## License

MIT
