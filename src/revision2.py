"""Round-2 revision analyses.

R9  - heterogeneity between the two external sources (Mendeley vs Kaggle/UCI)
      before pooling: per-feature PSI/KS, outcome prevalence, domain discriminator.
R10 - transfer to the Nigerian validation set from each external source alone,
      against the pooled development set.
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy import stats
from common import model, engineer, boot_auc, VITALS, SEED

df = engineer(pd.read_csv("data/processed/harmonized.csv"))
df["y"] = (df["risk3"] == "high").astype(int)
FULL = VITALS + ["bs", "pulse_pressure", "map", "shock_index"]

men = df[df.source == "mendeley"]
kag = df[df.source == "kaggle"]
ext = df[df.domain == "external"]
loc260 = df[df.source.isin(["first_mercy", "tim_unity"])]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def ece(y, p):
    return np.abs(np.subtract(*calibration_curve(y, p, n_bins=10, strategy="quantile"))).mean()

print("=" * 100)
print("R9  BETWEEN-SOURCE HETEROGENEITY WITHIN THE EXTERNAL POOL")
print(f"  Mendeley n={len(men)}  prevalence={men.y.mean():.3f}")
print(f"  Kaggle   n={len(kag)}  prevalence={kag.y.mean():.3f}")
ct = np.array([[men.y.sum(), (1 - men.y).sum()], [kag.y.sum(), (1 - kag.y).sum()]])
chi = stats.chi2_contingency(ct)
print(f"  Prevalence difference: {men.y.mean()-kag.y.mean():+.3f}  chi2 p={chi.pvalue:.2e}")

rows = []
print(f"\n  {'feature':16s} {'Mendeley':>10s} {'Kaggle':>10s} {'PSI':>7s} {'KS':>7s} {'KS p':>10s}")
for c in FULL:
    a, b = men[c].dropna(), kag[c].dropna()
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    ea = np.histogram(a, edges)[0] / len(a) + 1e-6
    eb = np.histogram(b, edges)[0] / len(b) + 1e-6
    psi = ((eb - ea) * np.log(eb / ea)).sum()
    ks = stats.ks_2samp(a, b)
    rows.append(dict(feature=c, mendeley_mean=a.mean(), kaggle_mean=b.mean(),
                     psi=psi, ks=ks.statistic, ks_p=ks.pvalue))
    print(f"  {c:16s} {a.mean():10.2f} {b.mean():10.2f} {psi:7.3f} {ks.statistic:7.3f} {ks.pvalue:10.2e}")
pd.DataFrame(rows).to_csv("outputs/R9_source_heterogeneity.csv", index=False)

dom = pd.concat([men.assign(d=0), kag.assign(d=1)])
dc = cross_val_predict(model(), dom[FULL], dom.d, cv=cv, method="predict_proba")[:, 1]
print(f"\n  Mendeley-vs-Kaggle domain discriminator AUROC = {roc_auc_score(dom.d, dc):.3f}")
print("  (external-vs-Nigerian discriminator was 0.991)")

print("\n" + "=" * 100)
print("R10 TRANSFER FROM EACH EXTERNAL SOURCE ALONE -> Nigerian validation set (n=260)")
out = []
for tag, tr in [("Mendeley only", men), ("Kaggle/UCI only", kag), ("Pooled external", ext)]:
    m = model().fit(tr[FULL], tr.y)
    p = m.predict_proba(loc260[FULL])[:, 1]
    y = loc260.y.values
    lo, hi = boot_auc(y, p)
    a, e, br = roc_auc_score(y, p), ece(y, p), brier_score_loss(y, p)
    pi = cross_val_predict(model(), tr[FULL], tr.y, cv=cv, method="predict_proba")[:, 1]
    ai, ei = roc_auc_score(tr.y, pi), ece(tr.y.values, pi)
    out.append(dict(training_source=tag, n_train=len(tr), internal_auroc=ai, internal_ece=ei,
                    external_auroc=a, ci_lo=lo, ci_hi=hi, ece=e, brier=br))
    print(f"  {tag:18s} n_train={len(tr):5d}  internal AUROC={ai:.3f} ECE={ei:.3f}  "
          f"-> Nigerian AUROC={a:.3f} [{lo:.3f}-{hi:.3f}] ECE={e:.3f} Brier={br:.3f}")
pd.DataFrame(out).to_csv("outputs/R10_source_transfer.csv", index=False)
print("\nWritten: outputs/R9_source_heterogeneity.csv, outputs/R10_source_transfer.csv")

print("\n" + "=" * 100)
print("R11 DISTANCE FROM EACH EXTERNAL SOURCE TO THE NIGERIAN VALIDATION SET (n=260)")
def psi(a, b):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    ea = np.histogram(a, edges)[0] / len(a) + 1e-6
    eb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return ((eb - ea) * np.log(eb / ea)).sum()

print(f"  {'feature':16s} {'Mendeley->NG':>13s} {'Kaggle->NG':>12s}  closer")
rows = []
for c in FULL:
    pm = psi(men[c].dropna(), loc260[c].dropna())
    pk = psi(kag[c].dropna(), loc260[c].dropna())
    rows.append(dict(feature=c, psi_mendeley_vs_ng=pm, psi_kaggle_vs_ng=pk))
    print(f"  {c:16s} {pm:13.3f} {pk:12.3f}  {'Kaggle' if pk < pm else 'Mendeley'}")
pd.DataFrame(rows).to_csv("outputs/R11_source_distance.csv", index=False)

for tag, src in [("Mendeley", men), ("Kaggle/UCI", kag)]:
    d = pd.concat([src.assign(d=0), loc260.assign(d=1)])
    dc = cross_val_predict(model(), d[FULL], d.d, cv=cv, method="predict_proba")[:, 1]
    print(f"  {tag:11s} vs Nigerian domain discriminator AUROC = {roc_auc_score(d.d, dc):.3f}"
          f"   | high-risk prevalence {src.y.mean():.3f} vs {loc260.y.mean():.3f}")
