#!/usr/bin/env python
# coding: utf-8

# 12_LASSO_Cox_Multiomics.py
# Second LASSO-Cox model: predictors = PAM50 mRNA expression + promoter methylation.
# Methylation is imputed fold-safe (KNN helper, fit on training patients only),
# then converted from beta to M-values (log2(beta/(1-beta))) for modelling, since
# M-values are more homoscedastic and better suited to linear models; expression is
# standardized per training fold. Uses the same shared 5-fold CV splits as the
# mRNA-only baseline (11) so the C-indices are comparable.
# High-dimensional L1-Cox is fit with scikit-survival's Coxnet (coordinate descent,
# suited to p >> n, unlike the lifelines fitter used for the 50-gene baseline).
# Run from inside scripts/ (relative paths, like the other Python scripts).

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from KNN_Imputation_Helper_Function import fit_transform_train_test_methylation

warnings.simplefilter("ignore")
Path("../results/tables").mkdir(parents=True, exist_ok=True)

# --- Load and align ----------------------------------------------------------
rna = pd.read_csv("../data/processed/rna_pam50.csv").set_index("patient")
meth = pd.read_csv("../data/processed/meth_pam50.csv").set_index("patient")  # raw, with NaNs
surv = pd.read_csv("../data/processed/survival_luminal_clean.csv").set_index("patient")
folds = pd.read_csv("../data/processed/cv_fold_assignments.csv").set_index("patient")

surv = surv[surv["time"].notna() & (surv["time"] > 0)]
patients = (rna.index.intersection(meth.index)
            .intersection(surv.index).intersection(folds.index))
rna, meth, surv = rna.loc[patients], meth.loc[patients], surv.loc[patients]
fold_id = folds.loc[patients, "fold"]
print(f"Patients: {len(patients)} | genes: {rna.shape[1]} | raw CpGs: {meth.shape[1]} "
      f"| folds: {sorted(fold_id.unique())}")

L1_RATIO = 1.0


def survival_y(ids):
    return Surv.from_arrays(event=surv.loc[ids, "event"].astype(bool).values,
                            time=surv.loc[ids, "time"].values)


def beta_to_m(B):
    """Convert methylation beta values to M-values: M = log2(beta / (1 - beta)).
    Beta is clipped away from 0/1 to keep M finite."""
    B = B.clip(1e-4, 1 - 1e-4)
    return np.log2(B / (1 - B))


def build_features(train_ids, test_ids):
    """Fold-safe predictors: per-fold KNN-imputed methylation converted to M-values
    and standardized, concatenated with per-fold standardized expression."""
    # Methylation: impute beta fold-safe, convert to M-values, standardize on train.
    Bm_tr, Bm_te, _ = fit_transform_train_test_methylation(meth, train_ids, test_ids, scale=False)
    Mm_tr, Mm_te = beta_to_m(Bm_tr), beta_to_m(Bm_te)
    m_scaler = StandardScaler().fit(Mm_tr)
    Mm_tr = pd.DataFrame(m_scaler.transform(Mm_tr), index=Mm_tr.index, columns=Mm_tr.columns)
    Mm_te = pd.DataFrame(m_scaler.transform(Mm_te), index=Mm_te.index, columns=Mm_te.columns)
    # Expression: standardize on train.
    r_scaler = StandardScaler().fit(rna.loc[train_ids])
    Xr_tr = pd.DataFrame(r_scaler.transform(rna.loc[train_ids]), index=train_ids, columns=rna.columns)
    Xr_te = pd.DataFrame(r_scaler.transform(rna.loc[test_ids]), index=test_ids, columns=rna.columns)
    X_tr = pd.concat([Xr_tr, Mm_tr], axis=1).to_numpy()
    X_te = pd.concat([Xr_te, Mm_te], axis=1).to_numpy()
    return X_tr, X_te


def select_alpha(X, y, alphas):
    """Inner 5-fold CV: fit the whole L1 path per inner-train fold and score each
    alpha on the inner-validation fold; return the alpha with the best mean C-index."""
    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    score_sum = np.zeros(len(alphas))
    for i_tr, i_va in inner.split(X):
        model = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alphas=alphas, max_iter=100000)
        try:
            model.fit(X[i_tr], y[i_tr])
        except (ArithmeticError, ValueError):  # Coxnet may fail to converge at small alphas
            continue
        for j, a in enumerate(alphas):
            risk = model.predict(X[i_va], alpha=a)
            score_sum[j] += concordance_index_censored(
                y[i_va]["event"], y[i_va]["time"], risk)[0]
    return alphas[int(np.argmax(score_sum))]


# --- Nested CV: outer = shared folds, inner = penalty selection ---------------
rows = []
for f in sorted(fold_id.unique()):
    train_ids = fold_id.index[fold_id != f]
    test_ids = fold_id.index[fold_id == f]
    X_tr, X_te = build_features(train_ids, test_ids)
    y_tr, y_te = survival_y(train_ids), survival_y(test_ids)

    # Data-driven penalty path from the training fold, then inner-CV selection.
    alphas = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, n_alphas=50,
                                    alpha_min_ratio=0.01, max_iter=100000).fit(X_tr, y_tr).alphas_
    best_alpha = select_alpha(X_tr, y_tr, alphas)

    final = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alphas=[best_alpha],
                                   max_iter=100000).fit(X_tr, y_tr)
    risk = final.predict(X_te)
    ci = concordance_index_censored(y_te["event"], y_te["time"], risk)[0]
    n_sel = int((final.coef_.ravel() != 0).sum())
    rows.append({"fold": f, "alpha": best_alpha, "n_features_total": X_tr.shape[1],
                 "n_features_selected": n_sel, "test_c_index": ci, "n_test": len(test_ids)})
    print(f"Fold {f}: C-index={ci:.3f} | alpha={best_alpha:.4g} | "
          f"features={X_tr.shape[1]} | selected={n_sel}")

cv = pd.DataFrame(rows)
cv.to_csv("../results/tables/lasso_cox_multiomics_cv_results.csv", index=False)

mean_ci, sd_ci = cv["test_c_index"].mean(), cv["test_c_index"].std()
print(f"\nMulti-omics LASSO-Cox C-index: {mean_ci:.3f} +/- {sd_ci:.3f} (5-fold CV)")
