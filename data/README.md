# Data

This repository contains **no patient data**.

## Public sources

Both public datasets must be obtained from their original repositories:

| Source | Records used | Where to obtain |
|---|---|---|
| Kaggle / UCI Maternal Health Risk | 1,270 raw (452 after de-duplication) | UCI Machine Learning Repository, "Maternal Health Risk Data Set". Originates with Ahmed & Kashem, doi: 10.1109/STI50764.2020.9350320 |
| Mendeley maternal health repository | 2,205 raw (1,072 after de-duplication) | Mendeley Data |

Place them at `data/raw/kaggle.csv` and `data/raw/mendeley.csv`, then run
`src/harmonize.py`.

## Nigerian clinical records

The 382 clinical records from the three southwestern Nigerian hospitals
(FUTH, First Mercy, Tim-Unity) are **not distributed here and cannot be made
public**. They are available from the corresponding author on reasonable request,
subject to permission from the contributing hospitals.

The arms of the study that depend on those records therefore cannot be
reproduced from this repository alone. Everything derived from them that does
not disclose individual records — the harmonisation audit trail, Table 1
summary statistics, all reported metrics and every figure — is included under
`outputs/`.

## A note on duplicates

1,921 of the 3,474 records in the two public repositories are exact
duplicates. `src/harmonize.py` removes them and writes the counts to
`outputs/audit_trail.csv`. Any pipeline that splits training and test data
before de-duplicating will leak identical rows across the split and report
inflated performance.
