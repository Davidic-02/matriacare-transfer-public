# When Benchmarks Do Not Travel

External validation and distributional-shift analysis of maternal risk prediction
models in Nigerian clinical populations.

External validation of a maternal-risk stacking ensemble trained on public
benchmarks (Mendeley, Kaggle/UCI) against clinical records from three
southwestern Nigerian hospitals.

## Reproducing

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
PYTHONPATH=src python src/harmonize.py            # raw -> data/processed/harmonized.csv + audit trail
PYTHONPATH=src python src/experiment.py   # Arms A0/A1/A2/B/R + shift diagnostics
PYTHONPATH=src python src/figures.py      # Figures 1-4, site breakdown, PSI table
PYTHONPATH=src python src/table1_flow_fairness.py # Table 1, Figure 5 flow, Figure 6 fairness
```

All randomness is seeded (`SEED = 42` in `src/common.py`). Runtime is a few
minutes on CPU.

## Layout

| Path | Contents |
|---|---|
| `data/` | How to obtain the public datasets; see `data/README.md`. No patient data is distributed in this repository. |

| `src/common.py` | Feature engineering, model definition, bootstrap metrics |
| `outputs/` | Figures, tables, audit trail, TRIPOD+AI checklist |

## Endpoint

High risk versus not-high risk — the one label definition all five sources
support. Mendeley and FUTH carry no mid-risk category.

## Known data issues

- 1,921 exact duplicates across Mendeley and Kaggle (removed; see `outputs/audit_trail.csv`)
- FUTH `Height` column mixes units (range 1.13-7.2); excluded from the shared schema
- Nigerian blood sugar is floored at exactly 6.0 mmol/L - probable recording
  threshold rather than physiology; pending confirmation from the sites
- Risk-label assignment protocol at each Nigerian site is not yet documented

## Data availability

This repository contains **no patient data**. The two public datasets must be
downloaded from their original repositories (see `data/README.md`); the
Nigerian clinical records are available only under institutional data-sharing
agreement with the contributing hospitals.

## Citation

Adekoya D., Akinbo R. S. "When Benchmarks Do Not Travel: External Validation
and Distributional Shift of Maternal Risk Prediction Models in Nigerian
Clinical Populations." Manuscript under review.
