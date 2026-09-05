"""Shared configuration, feature engineering, model definition and metrics."""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score
from sklearn.calibration import calibration_curve
from scipy import stats

SEED = 42
VITALS = ["age", "sbp", "dbp", "body_temp", "heart_rate"]

def engineer(d):
    d = d.copy()
    d["pulse_pressure"] = d["sbp"] - d["dbp"]
    d["map"] = d["dbp"] + (d["sbp"] - d["dbp"]) / 3
    d["shock_index"] = d["heart_rate"] / d["sbp"]
    return d

def model():
    base = [("rf", RandomForestClassifier(n_estimators=400, random_state=SEED)),
            ("gb", GradientBoostingClassifier(random_state=SEED)),
            ("svm", Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=SEED))]))]
    stack = StackingClassifier(base, final_estimator=LogisticRegression(max_iter=2000),
                               cv=5, stack_method="predict_proba")
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", stack)])

def boot_auc(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            s.append(roc_auc_score(y[i], p[i]))
    return np.percentile(s, [2.5, 97.5])

def report(tag, y, p):
    lo, hi = boot_auc(y, p)
    auc = roc_auc_score(y, p)
    # sensitivity at the threshold giving ~80% specificity, a triage-relevant operating point
    thr = np.quantile(p[y == 0], 0.80)
    sens = recall_score(y, (p >= thr).astype(int))
    ece = np.abs(np.subtract(*calibration_curve(y, p, n_bins=10, strategy="quantile"))).mean()
    print(f"{tag:38s} n={len(y):5d} prev={y.mean():.2f}  AUROC={auc:.3f} [{lo:.3f}-{hi:.3f}]  "
          f"AUPRC={average_precision_score(y,p):.3f}  Sens@80Spec={sens:.3f}  Brier={brier_score_loss(y,p):.3f}  ECE={ece:.3f}")
    return auc

