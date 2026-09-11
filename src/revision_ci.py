"""Uncertainty for calibration error and seed stability of the transfer result."""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from common import engineer, VITALS, SEED

FULL = VITALS + ["bs", "pulse_pressure", "map", "shock_index"]
df = engineer(pd.read_csv("data/processed/harmonized.csv"))
df["y"] = (df["risk3"] == "high").astype(int)
ext = df[df.domain == "external"]
loc = df[df.source.isin(["first_mercy", "tim_unity"])]

def ece(y, p):
    a, b = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return float(np.abs(a - b).mean())

def boot_ece(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1: s.append(ece(y[i], p[i]))
    return np.percentile(s, [2.5, 97.5])

def learners(seed):
    imp = ("i", SimpleImputer(strategy="median"))
    stack = StackingClassifier(
        [("rf", RandomForestClassifier(n_estimators=400, random_state=seed)),
         ("gb", GradientBoostingClassifier(random_state=seed)),
         ("svm", Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=seed))]))],
        final_estimator=LogisticRegression(max_iter=2000), cv=5, stack_method="predict_proba")
    return {
        "Logistic regression": Pipeline([imp, ("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]),
        "Random forest": Pipeline([imp, ("m", RandomForestClassifier(n_estimators=400, random_state=seed))]),
        "Gradient boosting": Pipeline([imp, ("m", GradientBoostingClassifier(random_state=seed))]),
        "SVM (RBF)": Pipeline([imp, ("s", StandardScaler()), ("m", SVC(probability=True, random_state=seed))]),
        "Stacking ensemble": Pipeline([imp, ("m", stack)]),
    }

# ---- ECE with bootstrap intervals, internal (out-of-fold) and external ----
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, mdl in learners(SEED).items():
    pi = cross_val_predict(mdl, ext[FULL], ext.y, cv=cv, method="predict_proba")[:, 1]
    pe = mdl.fit(ext[FULL], ext.y).predict_proba(loc[FULL])[:, 1]
    il, ih = boot_ece(ext.y.values, pi); el, eh = boot_ece(loc.y.values, pe)
    rows.append(dict(model=name, internal_ece=round(ece(ext.y.values, pi), 3),
                     int_lo=round(il, 3), int_hi=round(ih, 3),
                     external_ece=round(ece(loc.y.values, pe), 3),
                     ext_lo=round(el, 3), ext_hi=round(eh, 3)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("outputs/R7_ece_intervals.csv", index=False)

# ---- seed stability of the primary external arm ----
rows = []
for seed in range(5):
    m = learners(seed)["Stacking ensemble"].fit(ext[FULL], ext.y)
    p = m.predict_proba(loc[FULL])[:, 1]
    lr = learners(seed)["Logistic regression"].fit(ext[FULL], ext.y).predict_proba(loc[FULL])[:, 1]
    rows.append(dict(seed=seed, stack_auroc=round(roc_auc_score(loc.y, p), 3), stack_ece=round(ece(loc.y.values, p), 3),
                     lr_auroc=round(roc_auc_score(loc.y, lr), 3)))
    print(rows[-1], flush=True)
s = pd.DataFrame(rows); s.to_csv("outputs/R8_seed_stability.csv", index=False)
print("\nstack AUROC range:", s.stack_auroc.min(), "-", s.stack_auroc.max(),
      "| ECE range:", s.stack_ece.min(), "-", s.stack_ece.max())
