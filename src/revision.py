"""Analyses requested in the FAITH pre-review (comments 1-5)."""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from common import model, engineer, boot_auc, VITALS, SEED

OUT = "outputs/"
FULL = VITALS + ["bs", "pulse_pressure", "map", "shock_index"]
VIT  = VITALS + ["pulse_pressure", "map", "shock_index"]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

df = engineer(pd.read_csv("data/processed/harmonized.csv"))
df["y"] = (df["risk3"] == "high").astype(int)
ext    = df[df.domain == "external"]
loc260 = df[df.source.isin(["first_mercy", "tim_unity"])]
futh   = df[df.source == "futh"]

def ece(y, p, bins=10):
    a, b = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return float(np.abs(a - b).mean())

def row(tag, y, p):
    lo, hi = boot_auc(y, p, n=2000)
    return dict(analysis=tag, n=len(y), prevalence=round(float(y.mean()), 3),
                auroc=round(roc_auc_score(y, p), 3), ci_low=round(lo, 3),
                ci_high=round(hi, 3), ece=round(ece(y, p), 3))

# ---- Comment 1: feature availability matrix -------------------------------
avail = df.groupby("source")[VITALS + ["bs"]].apply(lambda g: (1 - g.isna().mean()).round(3))
avail.to_csv(OUT + "R1_feature_availability.csv")
print("== R1 feature availability (proportion non-missing) ==")
print(avail.to_string(), "\n")

# ---- Comment 2: A1 and A2 on an identical (vitals-only) feature set -------
rows = []
m_vit  = model().fit(ext[VIT], ext.y)
m_full = model().fit(ext[FULL], ext.y)
rows.append(row("A1 external->First Mercy+Tim-Unity, FULL features", loc260.y.values, m_full.predict_proba(loc260[FULL])[:, 1]))
rows.append(row("A1 external->First Mercy+Tim-Unity, VITALS only",  loc260.y.values, m_vit.predict_proba(loc260[VIT])[:, 1]))
rows.append(row("A2 external->FUTH, VITALS only",                    futh.y.values,   m_vit.predict_proba(futh[VIT])[:, 1]))
r2 = pd.DataFrame(rows); r2.to_csv(OUT + "R2_matched_featureset.csv", index=False)
print("== R2 matched feature set ==\n", r2.to_string(index=False), "\n")

# ---- Comment 3: sample-size-matched comparison ----------------------------
# Local ceiling (n=260) vs external models trained on repeated n=260 subsamples.
rng = np.random.default_rng(SEED)
p_local = cross_val_predict(model(), loc260[FULL], loc260.y, cv=cv, method="predict_proba")[:, 1]
local_auc = roc_auc_score(loc260.y, p_local)
sub = []
for i in range(25):
    idx = rng.choice(len(ext), 260, replace=False)
    e = ext.iloc[idx]
    if e.y.nunique() < 2: continue
    m = model().fit(e[FULL], e.y)
    sub.append(roc_auc_score(loc260.y, m.predict_proba(loc260[FULL])[:, 1]))
sub = np.array(sub)
full_ext = roc_auc_score(loc260.y, m_full.predict_proba(loc260[FULL])[:, 1])
r3 = pd.DataFrame([
    dict(comparison="Local model, 5-fold CV (n=260)", auroc=round(local_auc, 3), note="local ceiling"),
    dict(comparison="External model, full n=1,524",   auroc=round(full_ext, 3),  note="as reported"),
    dict(comparison="External model, n=260 subsamples (25 draws)",
         auroc=round(sub.mean(), 3), note=f"SD {sub.std():.3f}, range {sub.min():.3f}-{sub.max():.3f}"),
])
r3.to_csv(OUT + "R3_size_matched.csv", index=False)
print("== R3 size-matched ==\n", r3.to_string(index=False), "\n")

# ---- Comment 4: is degradation specific to the stacking model? ------------
learners = {
    "Logistic regression": Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=2000))]),
    "Random forest":       Pipeline([("i", SimpleImputer(strategy="median")),
                                     ("m", RandomForestClassifier(n_estimators=400, random_state=SEED))]),
    "Gradient boosting":   Pipeline([("i", SimpleImputer(strategy="median")),
                                     ("m", GradientBoostingClassifier(random_state=SEED))]),
    "SVM (RBF)":           Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                                     ("m", SVC(probability=True, random_state=SEED))]),
    "Stacking ensemble":   model(),
}
rows = []
for name, mdl in learners.items():
    pi = cross_val_predict(mdl, ext[FULL], ext.y, cv=cv, method="predict_proba")[:, 1]
    fitted = mdl.fit(ext[FULL], ext.y)
    pe = fitted.predict_proba(loc260[FULL])[:, 1]
    rows.append(dict(model=name,
                     internal_auroc=round(roc_auc_score(ext.y, pi), 3),
                     external_auroc=round(roc_auc_score(loc260.y, pe), 3),
                     delta_auroc=round(roc_auc_score(loc260.y, pe) - roc_auc_score(ext.y, pi), 3),
                     internal_ece=round(ece(ext.y.values, pi), 3),
                     external_ece=round(ece(loc260.y.values, pe), 3)))
r4 = pd.DataFrame(rows); r4.to_csv(OUT + "R4_model_agnostic.csv", index=False)
print("== R4 model-agnostic degradation ==\n", r4.to_string(index=False), "\n")

# ---- Comment 5: is the glucose-tertile collapse a glucose artefact? -------
p_full = m_full.predict_proba(loc260[FULL])[:, 1]
p_vit  = m_vit.predict_proba(loc260[VIT])[:, 1]
g = loc260.assign(p_full=p_full, p_vit=p_vit)
g["tertile"] = pd.qcut(g.bs, 3, labels=["low", "mid", "high"])
rows = []
for lvl, s in g.groupby("tertile", observed=True):
    for tag, col in [("model WITH glucose", "p_full"), ("model WITHOUT glucose", "p_vit")]:
        if s.y.nunique() < 2: continue
        lo, hi = boot_auc(s.y.values, s[col].values, n=2000)
        rows.append(dict(tertile=lvl, model=tag, n=len(s), prevalence=round(s.y.mean(), 3),
                         auroc=round(roc_auc_score(s.y, s[col]), 3), ci_low=round(lo, 3), ci_high=round(hi, 3)))
# excluding records sitting exactly on the 6.0 recording floor
nf = g[g.bs > 6.0]
if nf.y.nunique() > 1:
    lo, hi = boot_auc(nf.y.values, nf.p_full.values, n=2000)
    rows.append(dict(tertile="all, excluding bs=6.0 floor", model="model WITH glucose", n=len(nf),
                     prevalence=round(nf.y.mean(), 3), auroc=round(roc_auc_score(nf.y, nf.p_full), 3),
                     ci_low=round(lo, 3), ci_high=round(hi, 3)))
r5 = pd.DataFrame(rows); r5.to_csv(OUT + "R5_glucose_sensitivity.csv", index=False)
print("== R5 glucose sensitivity ==\n", r5.to_string(index=False), "\n")
print("records at exactly 6.0 mmol/L:", int((loc260.bs == 6.0).sum()), "of", len(loc260))

# ---- Comment 6: label prevalence across sources ---------------------------
lab = pd.crosstab(df.source, df.risk3, normalize="index").round(3)
lab["n"] = df.groupby("source").size()
lab.to_csv(OUT + "R6_label_distribution.csv")
print("\n== R6 label distribution ==\n", lab.to_string())
