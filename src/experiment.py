"""Arm A/B/C external-validation experiment plus domain-shift diagnostics.

Primary endpoint is the referral decision: high risk vs not high risk, the one
label definition every source supports.
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy import stats
from common import model, engineer, boot_auc, report, VITALS, SEED

df = engineer(pd.read_csv("data/processed/harmonized.csv"))
df["y"] = (df["risk3"] == "high").astype(int)

FULL = VITALS + ["bs", "pulse_pressure", "map", "shock_index"]

ext = df[df.domain == "external"]
loc260 = df[df.source.isin(["first_mercy", "tim_unity"])]
futh = df[df.source == "futh"]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

print("=" * 118)
print("ARM A - train external (Mendeley+Kaggle, deduplicated), test Nigerian records")
m = model().fit(ext[FULL], ext.y)
report("  A1 external -> local (n=260)", loc260.y.values, m.predict_proba(loc260[FULL])[:, 1])

mv = model().fit(ext[VITALS + ["pulse_pressure", "map", "shock_index"]], ext.y)
report("  A2 external -> FUTH (vitals only)", futh.y.values,
       mv.predict_proba(futh[VITALS + ["pulse_pressure", "map", "shock_index"]])[:, 1])

print("\nARM A0 - internal reference (what the external model scores on its own data)")
p = cross_val_predict(model(), ext[FULL], ext.y, cv=cv, method="predict_proba")[:, 1]
report("  A0 external 5-fold CV", ext.y.values, p)

print("\nARM B - local ceiling (5-fold CV within the 260 Nigerian records)")
p = cross_val_predict(model(), loc260[FULL], loc260.y, cv=cv, method="predict_proba")[:, 1]
report("  B  local CV", loc260.y.values, p)

print("\nREVERSE - train local (n=260), test external")
mr = model().fit(loc260[FULL], loc260.y)
report("  R  local -> external", ext.y.values, mr.predict_proba(ext[FULL])[:, 1])

print("\n" + "=" * 118)
print("DOMAIN SHIFT DIAGNOSTICS (external vs Nigerian 260)")
dom = pd.concat([ext.assign(d=0), loc260.assign(d=1)])
dc = cross_val_predict(model(), dom[FULL], dom.d, cv=cv, method="predict_proba")[:, 1]
print(f"  Domain-discriminator AUC = {roc_auc_score(dom.d, dc):.3f}  (0.5 = indistinguishable)")

print("\n  Per-feature shift:")
print(f"  {'feature':16s} {'ext mean':>10s} {'local mean':>11s} {'PSI':>7s} {'KS':>7s} {'KS p':>10s}")
for c in FULL:
    a, b = ext[c].dropna(), loc260[c].dropna()
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    ea = np.histogram(a, edges)[0] / len(a) + 1e-6
    eb = np.histogram(b, edges)[0] / len(b) + 1e-6
    psi = ((eb - ea) * np.log(eb / ea)).sum()
    ks = stats.ks_2samp(a, b)
    print(f"  {c:16s} {a.mean():10.2f} {b.mean():11.2f} {psi:7.3f} {ks.statistic:7.3f} {ks.pvalue:10.2e}")
