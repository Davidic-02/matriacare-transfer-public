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
five-fold. A model trained on 260 local records outperforms one trained on
1,524 external records. In the highest blood-sugar tertile, discrimination
collapses to 0.581 [0.430–0.732] — invisible in the pooled figure.

## Reproducing

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
PYTHONPATH=src python src/harmonize.py             # raw -> harmonised table + audit trail
PYTHONPATH=src python src/experiment.py            # transfer arms + shift diagnostics
PYTHONPATH=src python src/figures.py               # Figures 1-4, site breakdown, PSI table
PYTHONPATH=src python src/table1_flow_fairness.py  # Table 1, flow diagram, fairness analysis
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
| `outputs/` | Figures, result tables, audit trail, TRIPOD+AI checklist |
| `data/` | How to obtain the public datasets. No patient data is distributed here. |

## Endpoint

High risk versus not-high risk — the one label definition all five sources
support. The Mendeley and FUTH sources carry no mid-risk category.

## Known data issues

- 1,921 of 3,474 records in the two public repositories are exact duplicates.
  Removed here; counts in `outputs/audit_trail.csv`. Splitting train and test
  before de-duplicating will leak identical rows across the split.
- The FUTH `Height` column mixes units (range 1.13–7.2) and is excluded from
  the shared schema.
- Nigerian blood sugar is floored at exactly 6.0 mmol/L, which is more likely
  a recording threshold than physiology. Confirmation from the sites is pending.

## Data availability

This repository contains **no patient data**. The two public datasets must be
downloaded from their original repositories (see `data/README.md`). The
Nigerian clinical records are available only under an institutional
data-sharing agreement with the contributing hospitals.

## Citation

Adekoya D., Akinbo R. S. "When Benchmarks Do Not Travel: External Validation
and Distributional Shift of Maternal Risk Prediction Models in Nigerian
Clinical Populations." Manuscript under review.

## License

MIT
