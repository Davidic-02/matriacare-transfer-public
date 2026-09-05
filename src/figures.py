"""Refit the experiment arms, cache predictions, and render the manuscript figures."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, brier_score_loss, recall_score
from sklearn.calibration import calibration_curve
from scipy import stats
from common import model, engineer, VITALS, SEED, boot_auc

FIG = "outputs/"
plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})

df = engineer(pd.read_csv("data/processed/harmonized.csv"))
df["y"] = (df["risk3"] == "high").astype(int)
FULL = VITALS + ["bs", "pulse_pressure", "map", "shock_index"]
VIT = VITALS + ["pulse_pressure", "map", "shock_index"]

ext = df[df.domain == "external"]
loc260 = df[df.source.isin(["first_mercy", "tim_unity"])]
futh = df[df.source == "futh"]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

P = {}
m_full = model().fit(ext[FULL], ext.y)
P["A1 External → Nigerian (n=260)"] = (loc260.y.values, m_full.predict_proba(loc260[FULL])[:, 1])
m_vit = model().fit(ext[VIT], ext.y)
P["A2 External → FUTH (n=122)"] = (futh.y.values, m_vit.predict_proba(futh[VIT])[:, 1])
P["A0 External internal CV (n=1524)"] = (ext.y.values, cross_val_predict(model(), ext[FULL], ext.y, cv=cv, method="predict_proba")[:, 1])
P["B Local CV (n=260)"] = (loc260.y.values, cross_val_predict(model(), loc260[FULL], loc260.y, cv=cv, method="predict_proba")[:, 1])
m_rev = model().fit(loc260[FULL], loc260.y)
P["R Local → External (n=1524)"] = (ext.y.values, m_rev.predict_proba(ext[FULL])[:, 1])
np.savez(FIG + "predictions.npz", **{k: np.vstack(v) for k, v in P.items()})

ORDER = ["A0 External internal CV (n=1524)", "B Local CV (n=260)",
         "A1 External → Nigerian (n=260)", "R Local → External (n=1524)",
         "A2 External → FUTH (n=122)"]
COL = {ORDER[0]: "#1b4965", ORDER[1]: "#2a9d8f", ORDER[2]: "#e76f51", ORDER[3]: "#8d6a9f", ORDER[4]: "#b00020"}

# Figure 1 - ROC overlay
fig, ax = plt.subplots(figsize=(5.2, 5))
for k in ORDER:
    y, p = P[k]; fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=COL[k], lw=1.8, label=f"{k} — AUC {roc_auc_score(y,p):.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=.5)
ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", title="Discrimination across transfer directions")
ax.legend(fontsize=7, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(FIG + "Figure1_ROC.png"); plt.close(fig)

# Figure 2 - calibration
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
for k in ORDER:
    y, p = P[k]
    pt, pp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    a1.plot(pp, pt, "o-", ms=3.5, lw=1.5, color=COL[k], label=k)
a1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=.5)
a1.set(xlabel="Mean predicted probability", ylabel="Observed fraction high-risk", title="Calibration")
a1.legend(fontsize=7, frameon=False)
ece = {k: np.abs(np.subtract(*calibration_curve(*P[k][::-1][::-1], n_bins=10, strategy="quantile"))).mean() for k in ORDER}
ks = list(ece)
a2.barh(range(len(ks)), [ece[k] for k in ks], color=[COL[k] for k in ks])
a2.set_yticks(range(len(ks))); a2.set_yticklabels([k.split()[0] for k in ks])
a2.set(xlabel="Expected calibration error", title="Calibration error by arm")
for i, k in enumerate(ks):
    a2.text(ece[k] + .004, i, f"{ece[k]:.3f}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG + "Figure2_Calibration.png"); plt.close(fig)

# Figure 3 - distribution shift
psi_rows = []
for c in FULL:
    a, b = ext[c].dropna(), loc260[c].dropna()
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    ea = np.histogram(a, edges)[0] / len(a) + 1e-6
    eb = np.histogram(b, edges)[0] / len(b) + 1e-6
    psi_rows.append((c, ((eb - ea) * np.log(eb / ea)).sum(), stats.ks_2samp(a, b).statistic))
psi = pd.DataFrame(psi_rows, columns=["feature", "psi", "ks"]).sort_values("psi")
psi.to_csv(FIG + "psi_table.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
cols = ["#b00020" if v > 0.25 else "#4a7c8c" for v in psi.psi]
axes[0].barh(psi.feature, psi.psi, color=cols)
axes[0].axvline(0.25, ls="--", c="k", lw=.8)
axes[0].set(xlabel="Population stability index", title="Feature shift (dashed = major-shift threshold)")
for ax, c in zip(axes[1:], ["bs", "age"]):
    ax.hist(ext[c].dropna(), bins=30, density=True, alpha=.55, color="#1b4965", label="External")
    ax.hist(loc260[c].dropna(), bins=30, density=True, alpha=.55, color="#e76f51", label="Nigerian")
    ax.set(xlabel={"bs": "Blood sugar (mmol/L)", "age": "Age (years)"}[c], ylabel="Density",
           title=f"{c} — PSI {psi.set_index('feature').psi[c]:.2f}")
    ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(FIG + "Figure3_Shift.png"); plt.close(fig)

# Figure 4 + site table
rows = []
for name, g, mdl, feats in [("First Mercy", loc260[loc260.source == "first_mercy"], m_full, FULL),
                            ("Tim-Unity", loc260[loc260.source == "tim_unity"], m_full, FULL),
                            ("FUTH (vitals only)", futh, m_vit, VIT),
                            ("Pooled 260", loc260, m_full, FULL)]:
    p = mdl.predict_proba(g[feats])[:, 1]; y = g.y.values
    lo, hi = boot_auc(y, p, n=2000)
    thr = np.quantile(p[y == 0], 0.80) if (y == 0).sum() else np.nan
    rows.append(dict(site=name, n=len(y), prevalence=round(y.mean(), 3),
                     auroc=round(roc_auc_score(y, p), 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                     auprc=round(average_precision_score(y, p), 3),
                     sens_at_80spec=round(recall_score(y, (p >= thr).astype(int)), 3),
                     brier=round(brier_score_loss(y, p), 3)))
site = pd.DataFrame(rows); site.to_csv(FIG + "site_breakdown.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 3.6))
yy = np.arange(len(site))
ax.errorbar(site.auroc, yy, xerr=[site.auroc - site.ci_low, site.ci_high - site.auroc],
            fmt="o", color="#1b4965", capsize=3, ms=6)
ax.axvline(0.5, ls="--", c="grey", lw=.8)
ax.set_yticks(yy); ax.set_yticklabels([f"{r.site}\n(n={r.n}, prev={r.prevalence:.2f})" for r in site.itertuples()])
ax.set(xlabel="AUROC (95% bootstrap CI)", title="External-model performance by Nigerian site", xlim=(0.3, 1.02))
fig.tight_layout(); fig.savefig(FIG + "Figure4_Sites.png"); plt.close(fig)

print(site.to_string(index=False)); print(); print(psi.to_string(index=False))
print("\nECE:", {k.split()[0]: round(v, 3) for k, v in ece.items()})
