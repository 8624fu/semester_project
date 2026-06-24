#!/usr/bin/env python
# coding: utf-8

# 11_LASSO_Cox_mRNA_Baseline.py
# Task 2: mRNA-only LASSO-Cox baseline for overall survival.
# Establishes the benchmark C-index that the later integrated (meth + mRNA)
# model must beat. Uses the shared 5-fold CV splits so all models are comparable.
# Run from inside scripts/ (relative paths, like the other Python scripts).

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

Path("../results/tables").mkdir(parents=True, exist_ok=True)

# --- Load and align ----------------------------------------------------------
rna = pd.read_csv("../data/processed/rna_pam50.csv").set_index("patient")
surv = pd.read_csv("../data/processed/survival_luminal_clean.csv").set_index("patient")
folds = pd.read_csv("../data/processed/cv_fold_assignments.csv").set_index("patient")

surv = surv[surv["time"].notna() & (surv["time"] > 0)]
patients = rna.index.intersection(surv.index).intersection(folds.index)
rna, surv, folds = rna.loc[patients], surv.loc[patients], folds.loc[patients]
fold_id = folds["fold"]  # shared 5-fold CV assignment (values 1..5)
print(f"Patients: {len(patients)} | genes: {rna.shape[1]} | folds: {sorted(fold_id.unique())}")

genes = list(rna.columns)
L1_RATIO = 1.0                            # pure LASSO
PENALTIES = [0.001, 0.005, 0.01, 0.05]    # inner-CV grid


def fit_cox(X, T, E, penalizer):
    """Fit a LASSO-penalized Cox model on a scaled design matrix."""
    df = X.copy()
    df["time"], df["event"] = T.values, E.values
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=L1_RATIO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="time", event_col="event")
    return cph


def c_index(cph, X, T, E):
    # Higher partial hazard = higher risk; concordance_index wants risk negated.
    risk = cph.predict_partial_hazard(X)
    return concordance_index(T, -risk, E)


# --- Nested CV: outer = shared folds, inner = penalty selection ---------------
rows = []
for f in sorted(fold_id.unique()):
    tr, te = fold_id != f, fold_id == f
    X_tr, X_te = rna[tr], rna[te]
    T_tr, E_tr = surv.loc[tr, "time"], surv.loc[tr, "event"]
    T_te, E_te = surv.loc[te, "time"], surv.loc[te, "event"]

    # Scale on training patients only (fold-safe, no leakage).
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), index=X_tr.index, columns=genes)
    X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=genes)

    # Inner 3-fold CV on the training set to pick the penalty.
    inner = KFold(n_splits=3, shuffle=True, random_state=42)
    best_pen, best_ci = PENALTIES[0], -np.inf
    for pen in PENALTIES:
        scores = []
        for i_tr, i_va in inner.split(X_tr_s):
            cph = fit_cox(X_tr_s.iloc[i_tr], T_tr.iloc[i_tr], E_tr.iloc[i_tr], pen)
            scores.append(c_index(cph, X_tr_s.iloc[i_va], T_tr.iloc[i_va], E_tr.iloc[i_va]))
        if np.mean(scores) > best_ci:
            best_ci, best_pen = np.mean(scores), pen

    # Refit on the full outer-train with the chosen penalty, evaluate on outer-test.
    cph = fit_cox(X_tr_s, T_tr, E_tr, best_pen)
    ci = c_index(cph, X_te_s, T_te, E_te)
    n_sel = int((cph.params_.abs() > 1e-6).sum())
    rows.append({"fold": f, "penalizer": best_pen, "n_features_selected": n_sel,
                 "test_c_index": ci, "n_test": int(te.sum())})
    print(f"Fold {f}: C-index={ci:.3f} | penalizer={best_pen} | genes kept={n_sel}")

cv = pd.DataFrame(rows)
cv.to_csv("../results/tables/lasso_cox_cv_results.csv", index=False)

mean_ci, sd_ci = cv["test_c_index"].mean(), cv["test_c_index"].std()
print(f"\nBenchmark mRNA-only LASSO-Cox C-index: {mean_ci:.3f} +/- {sd_ci:.3f} (5-fold CV)")
