"""Harmonize the five MATRIACARE sources into one auditable analysis table.

Every exclusion is counted and written to outputs/audit_trail.csv so the
numbers can be reproduced in a supplementary table.
"""
import pandas as pd, numpy as np, pathlib

RAW = pathlib.Path("data/raw"); OUT = pathlib.Path("data/processed")
REPORT = pathlib.Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)

# Physiologic plausibility bounds for a pregnant population. Values outside
# these are set missing rather than dropped, so a record is not lost for one
# bad field.
BOUNDS = {"age": (10, 60), "sbp": (70, 200), "dbp": (40, 140),
          "body_temp": (95, 106), "heart_rate": (40, 160), "bs": (3, 30)}

SHARED = ["age", "sbp", "dbp", "body_temp", "heart_rate", "bs"]

def norm_label(s):
    s = str(s).strip().lower().replace(" risk", "").replace("risk", "").strip()
    return {"high": "high", "mid": "mid", "medium": "mid", "low": "low"}.get(s, np.nan)

def load():
    d = {}
    f = pd.read_csv(RAW / "futh.csv")
    d["futh"] = f.rename(columns={"Age": "age", "StytolicBp": "sbp", "DiastolicBp": "dbp",
                                  "BodyTemp": "body_temp", "Heart Rate": "heart_rate",
                                  "RiskLevel": "risk_raw"}).assign(bs=np.nan)

    k = pd.read_csv(RAW / "kaggle.csv")
    k.columns = [c.strip().lstrip("﻿") for c in k.columns]
    d["kaggle"] = k.rename(columns={"Age": "age", "SystolicBP": "sbp", "DiastolicBP": "dbp",
                                    "BS": "bs", "BodyTemp": "body_temp",
                                    "HeartRate": "heart_rate", "RiskLevel": "risk_raw"})

    m = pd.read_csv(RAW / "mendeley.csv")
    d["mendeley"] = m.rename(columns={"Age": "age", "Systolic BP": "sbp", "Diastolic": "dbp",
                                      "BS": "bs", "Body Temp": "body_temp",
                                      "Heart Rate": "heart_rate", "Risk Level": "risk_raw"})

    fm = pd.read_excel(RAW / "first_mercy.xlsx", header=1).dropna(how="all")
    d["first_mercy"] = fm.rename(columns={"Age": "age", "SystolicBP": "sbp", "DiastolicBP": "dbp",
                                          "BS": "bs", "BodyTemp": "body_temp",
                                          "HeartRate": "heart_rate", "RiskLevel": "risk_raw"})

    tu = pd.read_excel(RAW / "tim_unity.xlsx", header=2).dropna(how="all")
    d["tim_unity"] = tu.rename(columns={"Age": "age", "Systolic BP": "sbp", "Diastolic BP": "dbp",
                                        "Blood Sugar": "bs", "Body Temp (°F)": "body_temp",
                                        "Heart Rate": "heart_rate", "Risk Level": "risk_raw"})
    return d

rows, frames = [], []
for name, df in load().items():
    n0 = len(df)
    df = df[[c for c in SHARED + ["risk_raw"] if c in df.columns]].copy()
    for c in SHARED:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    implausible = {c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in BOUNDS.items()}
    for c, (lo, hi) in BOUNDS.items():
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

    df["risk3"] = df["risk_raw"].map(norm_label)
    n_nolabel = int(df["risk3"].isna().sum())
    df = df.dropna(subset=["risk3"])

    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df["source"] = name
    df["domain"] = "local" if name in ("futh", "first_mercy", "tim_unity") else "external"
    frames.append(df)
    rows.append(dict(source=name, raw_rows=n0, unlabelled=n_nolabel, duplicates=n_dupes,
                     final=len(df), **{f"implausible_{k}": v for k, v in implausible.items()}))

audit = pd.DataFrame(rows)
out = pd.concat(frames, ignore_index=True)
out.to_csv(OUT / "harmonized.csv", index=False)
audit.to_csv(REPORT / "audit_trail.csv", index=False)

print(audit.to_string(index=False))
print("\nTotal analysable records:", len(out))
print("\nLabel distribution by source:")
print(pd.crosstab(out["source"], out["risk3"], normalize="index").round(3).to_string())
print("\nCounts:"); print(pd.crosstab(out["source"], out["risk3"]).to_string())
print("\nMissingness by source (%):")
print((out.groupby("source")[SHARED].apply(lambda g: g.isna().mean() * 100)).round(1).to_string())
