"""Table 1, participant-flow diagram, and subgroup fairness analysis."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_auc_score, recall_score
from common import model, engineer, VITALS, SEED, boot_auc

OUT = "outputs/"
plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "font.family": "DejaVu Sans"})
FULL = VITALS + ["bs", "pulse_pressure", "map", "shock_index"]
LABEL = {"age": "Age, years", "sbp": "Systolic BP, mmHg", "dbp": "Diastolic BP, mmHg",
         "body_temp": "Body temperature, °F", "heart_rate": "Heart rate, bpm",
         "bs": "Blood sugar, mmol/L"}
NAME = {"futh": "FUTH (tertiary)", "first_mercy": "First Mercy (private)",
        "tim_unity": "Tim-Unity (specialist)", "mendeley": "Mendeley", "kaggle": "Kaggle/UCI"}

df = engineer(pd.read_csv("data/processed/harmonized.csv"))
df["y"] = (df["risk3"] == "high").astype(int)

# ---------------------------------------------------------------- Table 1
def col(g):
    out = {"n": len(g)}
    for c in LABEL:
        s = g[c].dropna()
        out[LABEL[c]] = "not measured" if s.empty else f"{s.mean():.1f} ({s.std():.1f})"
        miss = g[c].isna().mean() * 100
        out[LABEL[c]] += "" if s.empty or miss == 0 else f" [{miss:.1f}% miss]"
    for lv in ["high", "mid", "low"]:
        n = int((g.risk3 == lv).sum())
        out[f"{lv.capitalize()} risk, n (%)"] = f"{n} ({n/len(g)*100:.1f})"
    return out

groups = [(NAME[s], df[df.source == s]) for s in ["futh", "first_mercy", "tim_unity", "mendeley", "kaggle"]]
groups += [("All Nigerian", df[df.domain == "local"]), ("All external", df[df.domain == "external"])]
t1 = pd.DataFrame({n: col(g) for n, g in groups})
t1.to_csv(OUT + "Table1_baseline.csv")
print("TABLE 1 — Baseline characteristics, mean (SD)\n")
print(t1.to_string())

# ---------------------------------------------------- Participant flow diagram
audit = pd.read_csv(OUT + "audit_trail.csv").set_index("source")
raw_ext = int(audit.loc[["mendeley", "kaggle"], "raw_rows"].sum())
raw_loc = int(audit.loc[["futh", "first_mercy", "tim_unity"], "raw_rows"].sum())
ex = lambda ss, c: int(audit.loc[ss, c].sum())

fig, ax = plt.subplots(figsize=(9.5, 7.2)); ax.axis("off")
def box(x, y, w, h, txt, fc="#eef3f6", ec="#1b4965"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=8.5)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color="#1b4965", lw=1.1))

box(0.03, 0.86, 0.42, 0.11, f"External public repositories\nMendeley + Kaggle/UCI\nn = {raw_ext:,} records")
box(0.55, 0.86, 0.42, 0.11, f"Nigerian clinical sites (3)\nFUTH, First Mercy, Tim-Unity\nn = {raw_loc:,} records")
box(0.03, 0.64, 0.42, 0.11, f"Excluded\n• unlabelled: {ex(['mendeley','kaggle'],'unlabelled')}\n"
                            f"• exact duplicates: {ex(['mendeley','kaggle'],'duplicates'):,}", fc="#fbeee9", ec="#b00020")
box(0.55, 0.64, 0.42, 0.11, f"Excluded\n• unlabelled: {ex(['futh','first_mercy','tim_unity'],'unlabelled')}\n"
                            f"• exact duplicates: {ex(['futh','first_mercy','tim_unity'],'duplicates')}", fc="#fbeee9", ec="#b00020")
box(0.03, 0.42, 0.42, 0.11, f"Development set\nn = {int(audit.loc[['mendeley','kaggle'],'final'].sum()):,}\n"
                            f"(Mendeley {int(audit.loc['mendeley','final']):,} + Kaggle {int(audit.loc['kaggle','final'])})")
box(0.55, 0.42, 0.42, 0.11, f"Nigerian records retained\nn = {int(audit.loc[['futh','first_mercy','tim_unity'],'final'])}"
                            if False else f"Nigerian records retained\nn = {int(audit.loc[['futh','first_mercy','tim_unity'],'final'].sum())}")
box(0.55, 0.20, 0.20, 0.11, f"Primary validation\nFirst Mercy + Tim-Unity\nn = 260", fc="#e8f4f1", ec="#2a9d8f")
box(0.78, 0.20, 0.19, 0.11, f"Sensitivity\nFUTH\nn = 122\n(87% high risk,\nno blood sugar)", fc="#fbeee9", ec="#b00020")
box(0.03, 0.20, 0.42, 0.11, "Physiologic bounds applied to all sets;\nout-of-range values set missing\n(counts in audit_trail.csv)", fc="#f5f5f5", ec="#777")

for x in (0.24, 0.76):
    arrow(x, 0.86, x, 0.75); arrow(x, 0.64, x, 0.53)
arrow(0.24, 0.42, 0.24, 0.31)
arrow(0.70, 0.42, 0.65, 0.31); arrow(0.82, 0.42, 0.87, 0.31)
ax.set(xlim=(0, 1), ylim=(0.15, 1.0))
ax.set_title("Participant flow", fontsize=11, pad=4)
fig.tight_layout(); fig.savefig(OUT + "Figure5_Flow.png"); plt.close(fig)

# -------------------------------------------------------------- Fairness
ext = df[df.domain == "external"]; loc260 = df[df.source.isin(["first_mercy", "tim_unity"])]
m = model().fit(ext[FULL], ext.y)
p = m.predict_proba(loc260[FULL])[:, 1]
y = loc260.y.values

# One global operating point, as a deployed triage tool would use.
thr = np.quantile(p[y == 0], 0.80)
g = loc260.assign(p=p, pred=(p >= thr).astype(int))
g["age_band"] = pd.cut(g.age, [0, 24, 34, 200], labels=["<25", "25-34", "≥35"])
g["bs_tertile"] = pd.qcut(g.bs, 3, labels=["low", "mid", "high"])
g["site"] = g.source.map(NAME)

rows = []
for var in ["age_band", "bs_tertile", "site"]:
    for lvl, sub in g.groupby(var, observed=True):
        yy, pp, pr = sub.y.values, sub.p.values, sub.pred.values
        tpr = recall_score(yy, pr) if yy.sum() else np.nan
        fpr = pr[yy == 0].mean() if (yy == 0).sum() else np.nan
        auc = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
        lo, hi = boot_auc(yy, pp, n=2000) if len(np.unique(yy)) > 1 else (np.nan, np.nan)
        rows.append(dict(variable=var, subgroup=str(lvl), n=len(sub), prevalence=round(yy.mean(), 3),
                         auroc=round(auc, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                         tpr=round(tpr, 3), fpr=round(fpr, 3)))
fair = pd.DataFrame(rows)
fair.to_csv(OUT + "fairness_subgroups.csv", index=False)
print("\n\nFAIRNESS — one global threshold (80% specificity on the pooled 260)\n")
print(fair.to_string(index=False))
print("\nEqualised-odds gaps (max - min across subgroups):")
for var, sub in fair.groupby("variable"):
    print(f"  {var:12s} TPR gap = {sub.tpr.max()-sub.tpr.min():.3f}   FPR gap = {sub.fpr.max()-sub.fpr.min():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
for ax, (var, sub) in zip(axes, fair.groupby("variable")):
    yy = np.arange(len(sub))
    ax.errorbar(sub.auroc, yy, xerr=[sub.auroc - sub.ci_low, sub.ci_high - sub.auroc],
                fmt="o", color="#1b4965", capsize=3, ms=5)
    ax.axvline(0.865, ls="--", c="#e76f51", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([f"{r.subgroup} (n={r.n})" for r in sub.itertuples()])
    ax.set(xlabel="AUROC (95% CI)", title={"age_band": "Age band", "bs_tertile": "Blood-sugar tertile",
                                           "site": "Site"}[var], xlim=(0.4, 1.02))
axes[0].text(0.44, -0.7, "dashed = pooled AUROC 0.865", fontsize=7, color="#e76f51")
fig.tight_layout(); fig.savefig(OUT + "Figure6_Fairness.png"); plt.close(fig)
