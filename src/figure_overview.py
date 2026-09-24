"""Methodological overview figure (study workflow) - Figure 1."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "font.family": "DejaVu Sans"})

SRC, PROC, EVAL, SENS = "#dce6f1", "#e8e8e8", "#d6e9d6", "#f6e7cf"
EDGE = "#555555"

fig, ax = plt.subplots(figsize=(7.2, 8.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")

def box(x, y, w, h, title, body="", fc=PROC, ts=8.2, bs=7.2, tp=0.66, bp=0.30):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                fc=fc, ec=EDGE, lw=0.8))
    if body:
        ax.text(x + w / 2, y + h * tp, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", linespacing=1.3)
        ax.text(x + w / 2, y + h * bp, body, ha="center", va="center", fontsize=bs, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", linespacing=1.45)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color=EDGE, shrinkA=0, shrinkB=0))

def stage(y, n):
    ax.text(2.0, y, n, ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", bbox=dict(boxstyle="circle,pad=0.34", fc="#4a4a4a", ec="none"))

# 1 - data sources
stage(111, "1")
box(8, 106, 43, 10, "External benchmark sources",
    "Mendeley  n = 2,205\nKaggle / UCI  n = 1,269", fc=SRC)
box(56, 106, 38, 10, "Nigerian clinical sites",
    "FUTH 188 · First Mercy 162\nTim-Unity 105", fc=SRC)
ax.text(50, 117.4, "Five data sources — 3,929 records", ha="center", fontsize=8.6, fontweight="bold")

# 2 - harmonisation
arrow(29.5, 105.5, 29.5, 99.5); arrow(75, 105.5, 75, 99.5)
stage(94, "2")
box(8, 88, 86, 11, "Harmonisation and quality control",
    "variable mapping → common schema (6 shared predictors) → unit checks →\n"
    "plausibility bounds (out-of-range set to missing) → unlabelled records removed →\n"
    "identical rows removed → training-fold median imputation → label binarised to high vs not-high risk",
    fc=PROC, bs=7.0)

# 3 - analysis sets
arrow(50, 87.5, 50, 81.5)
stage(76, "3")
box(8, 70, 43, 11, "External development set",
    "n = 1,524 (Mendeley + Kaggle/UCI)\npooling assessed in §4.6", fc=SRC)
box(56, 70, 38, 11, "Nigerian validation sets",
    "n = 260 primary (First Mercy,\nTim-Unity) · n = 122 FUTH", fc=SRC)

# 4 - features + model
arrow(29.5, 69.5, 29.5, 63.5)
stage(58, "4")
box(8, 52, 86, 11, "Feature engineering and model development",
    "6 shared predictors (age, systolic BP, diastolic BP, temperature, heart rate, blood glucose)\n"
    "+ 3 derived (pulse pressure, mean arterial pressure, shock index)\n"
    "stacking ensemble: random forest · gradient boosting · SVM → logistic-regression meta-learner")
ax.plot([75, 97.5, 97.5], [69.5, 69.5, 40], color=EDGE, lw=0.9, solid_capstyle="round")
arrow(97.5, 40, 95.0, 40)

# 5 - validation arms
arrow(50, 51.5, 50, 45.5)
stage(40, "5")
box(8, 34, 86, 11, "Internal and external validation — five pre-specified arms",
    "A0 internal 5-fold CV  ·  A1 external → Nigerian (primary)  ·  A2 external → FUTH\n"
    "B within-population 5-fold CV  ·  R reverse transfer (Nigerian → external)", fc=EVAL)

# 6 - three evaluation strands
arrow(24, 33.5, 24, 28.5); arrow(50, 33.5, 50, 28.5); arrow(76, 33.5, 76, 28.5)
stage(22.5, "6")
box(6, 17, 28, 11, "Distributional shift",
    "population stability index\nKolmogorov–Smirnov\ndomain discriminator", fc=EVAL, bs=6.9, tp=0.80, bp=0.32)
box(36, 17, 28, 11, "Discrimination and calibration",
    "AUROC · AUPRC · Brier\nexpected calibration error\nreliability curves", fc=EVAL, ts=7.6, bs=6.9, tp=0.80, bp=0.32)
box(66, 17, 28, 11, "Subgroup analysis",
    "single global threshold\n(80% specificity)\nequalised-odds gaps", fc=EVAL, bs=6.9, tp=0.80, bp=0.32)

# 7 - sensitivity
arrow(24, 16.5, 24, 12.5); arrow(50, 16.5, 50, 12.5); arrow(76, 16.5, 76, 12.5)
stage(7.5, "7")
box(8, 2, 86, 10, "Sensitivity and robustness analyses",
    "matched (vitals-only) feature set  ·  size-matched external training  ·  alternative model classes\n"
    "model without blood glucose  ·  five additional random seeds  ·  source-specific transfer",
    fc=SENS, bs=7.0)

fig.savefig("outputs/Figure0_Overview.png", bbox_inches="tight", facecolor="white")
print("written outputs/Figure0_Overview.png")
