import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from lifelines.statistics import logrank_test

sys.path.append("../scripts")
from KNN_Imputation_Helper_Function import fit_transform_train_test_methylation

warnings.simplefilter("ignore")
TABLES = Path("../results/tables"); TABLES.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# --- load ---
surv  = pd.read_csv("../data/processed/survival_luminal_clean.csv").set_index("patient")
stage = pd.read_csv("../data/processed/stage_luminal_brca.csv").set_index("patient")
folds = pd.read_csv("../data/processed/cv_fold_assignments.csv").set_index("patient")
rna   = pd.read_csv("../data/processed/rna_pam50.csv").set_index("patient")
meth  = pd.read_csv("../data/processed/meth_pam50.csv").set_index("patient")
anno  = pd.read_csv("../data/processed/cpg_gene_map.csv")
cpg_to_gene = dict(zip(anno["cpg"], anno["gene"]))

surv = surv[surv["time"].notna() & (surv["time"] > 0)]
patients = surv.index.intersection(folds.index).intersection(rna.index).intersection(meth.index)
surv, rna, meth = surv.loc[patients], rna.loc[patients], meth.loc[patients]
fold_id = folds.loc[patients, "fold"]
print(f"Patients: {len(patients)} | events: {int(surv['event'].sum())} | folds: {sorted(fold_id.unique())}")

def survival_y(ids):
    return Surv.from_arrays(event=surv.loc[ids, "event"].astype(bool).values,
                            time=surv.loc[ids, "time"].values)

def cox_cv(X, label, alpha=0.1):
    """Unpenalized Cox (small ridge alpha for stability) over the shared folds; returns per-fold C-index."""
    cis = []
    for f in sorted(fold_id.unique()):
        tr = fold_id.index[fold_id != f]; te = fold_id.index[fold_id == f]
        m = CoxPHSurvivalAnalysis(alpha=alpha).fit(X.loc[tr].values, survival_y(tr))
        risk = m.predict(X.loc[te].values)
        y_te = survival_y(te)
        cis.append(concordance_index_censored(y_te["event"], y_te["time"], risk)[0])
        print(f"  [{label}] fold {f}: C-index={cis[-1]:.3f}")
    print(f"  [{label}] mean C-index = {np.mean(cis):.3f} +/- {np.std(cis):.3f}")
    return np.array(cis)

stage_g = stage["stage_grouped"].reindex(patients).fillna("Unknown")
X_clin = pd.get_dummies(stage_g, prefix="stage", drop_first=True).astype(float)
X_clin["subtype_LumB"] = (surv["BRCA_Subtype_PAM50"] == "LumB").astype(float)
X_clin.index = patients
print("Clinical design columns:", list(X_clin.columns))

ci_clin = cox_cv(X_clin, "clinical")

L1_RATIO = 0.5

def build_mrna(tr, te):
    sc = StandardScaler().fit(rna.loc[tr])
    return sc.transform(rna.loc[tr]), sc.transform(rna.loc[te])

def beta_to_m(B):
    B = B.clip(1e-4, 1 - 1e-4)
    return np.log2(B / (1 - B))

def build_multi(tr, te):
    Bm_tr, Bm_te, _ = fit_transform_train_test_methylation(meth, tr, te, scale=False)
    Mm_tr, Mm_te = beta_to_m(Bm_tr), beta_to_m(Bm_te)
    ms = StandardScaler().fit(Mm_tr)
    Mm_tr = pd.DataFrame(ms.transform(Mm_tr), index=Mm_tr.index, columns=Mm_tr.columns)
    Mm_te = pd.DataFrame(ms.transform(Mm_te), index=Mm_te.index, columns=Mm_te.columns)
    rs = StandardScaler().fit(rna.loc[tr])
    Xr_tr = pd.DataFrame(rs.transform(rna.loc[tr]), index=tr, columns=rna.columns)
    Xr_te = pd.DataFrame(rs.transform(rna.loc[te]), index=te, columns=rna.columns)
    return pd.concat([Xr_tr, Mm_tr], axis=1).to_numpy(), pd.concat([Xr_te, Mm_te], axis=1).to_numpy()

def select_alpha_min(X, y, alphas):
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = np.full((5, len(alphas)), np.nan)
    for k, (i_tr, i_va) in enumerate(inner.split(X, y["event"])):
        mdl = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alphas=alphas, max_iter=100000)
        try:
            mdl.fit(X[i_tr], y[i_tr])
            for j, a in enumerate(alphas):
                r = mdl.predict(X[i_va], alpha=a)
                scores[k, j] = concordance_index_censored(y[i_va]["event"], y[i_va]["time"], r)[0]
        except (ArithmeticError, ValueError):
            continue
    return alphas[np.nanargmax(np.nanmean(scores, axis=0))]

def elasticnet_cv(build_features, label, alpha_min_ratio):
    cis = []
    for f in sorted(fold_id.unique()):
        tr = fold_id.index[fold_id != f]; te = fold_id.index[fold_id == f]
        X_tr, X_te = build_features(tr, te)
        y_tr, y_te = survival_y(tr), survival_y(te)
        alphas = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, n_alphas=100,
                                        alpha_min_ratio=alpha_min_ratio,
                                        max_iter=100000).fit(X_tr, y_tr).alphas_
        a = select_alpha_min(X_tr, y_tr, alphas)
        for cand in np.sort(alphas)[np.sort(alphas) >= a]:
            try:
                final = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alphas=[cand], max_iter=100000).fit(X_tr, y_tr)
                break
            except ArithmeticError:
                continue
        ci = concordance_index_censored(y_te["event"], y_te["time"], final.predict(X_te))[0]
        cis.append(ci)
        print(f"  [{label}] fold {f}: C-index={ci:.3f}")
    print(f"  [{label}] mean C-index = {np.mean(cis):.3f} +/- {np.std(cis):.3f}")
    return np.array(cis)

ci_en_mrna  = elasticnet_cv(build_mrna,  "EN mRNA",       alpha_min_ratio=0.01)
ci_en_multi = elasticnet_cv(build_multi, "EN multi-omics", alpha_min_ratio=0.05)

X_sub = pd.DataFrame({"subtype_LumB": (surv["BRCA_Subtype_PAM50"] == "LumB").astype(float)},
                     index=patients)
ci_sub = cox_cv(X_sub, "subtype-only")

# log-rank LumA vs LumB on the full cohort
t = surv["time"] / 365.25
is_b = surv["BRCA_Subtype_PAM50"] == "LumB"
lr = logrank_test(t[~is_b], t[is_b],
                  event_observed_A=surv["event"][~is_b], event_observed_B=surv["event"][is_b])
print(f"\nLumA vs LumB log-rank p = {lr.p_value:.4g}  (LumA n={(~is_b).sum()}, LumB n={is_b.sum()})")

summary = pd.DataFrame([
    {"model": "Subtype only (LumA/LumB)", "mean_c_index": ci_sub.mean(),  "sd": ci_sub.std()},
    {"model": "Clinical (subtype + stage)", "mean_c_index": ci_clin.mean(), "sd": ci_clin.std()},
    {"model": "Elastic net - mRNA",        "mean_c_index": ci_en_mrna.mean(),  "sd": ci_en_mrna.std()},
    {"model": "Elastic net - multi-omics", "mean_c_index": ci_en_multi.mean(), "sd": ci_en_multi.std()},
]).round(3)
summary.to_csv(TABLES / "feedback_baseline_models.csv", index=False)
print(summary.to_string(index=False))
print("\n(Compare against the existing LASSO ~0.55-0.58 and NN ~0.62-0.64 models.)")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

FIGURES = Path("../results/figures"); FIGURES.mkdir(parents=True, exist_ok=True)

# original 4 models: per-fold test C-index from the first model comparison
orig_files = {
    "LASSO mRNA":        "lasso_cox_cv_results.csv",
    "LASSO multi-omics": "lasso_cox_multiomics_cv_results.csv",
    "NN mRNA":           "nn_mRNA_only_best_model_folds.csv",
    "NN integrated":     "nn_integrated_best_model_folds.csv",
}
orig = {name: pd.read_csv(TABLES / f)["test_c_index"].values for name, f in orig_files.items()}


data = {
    "Subtype only":              ci_sub,
    "Clinical\n(subtype+stage)": ci_clin,
    "LASSO mRNA":                orig["LASSO mRNA"],
    "Elastic net mRNA":          ci_en_mrna,
    "LASSO multi-omics":         orig["LASSO multi-omics"],
    "Elastic net multi-omics":   ci_en_multi,
    "NN mRNA":                   orig["NN mRNA"],
    "NN integrated":             orig["NN integrated"],
}
FEEDBACK = {"Subtype only", "Clinical\n(subtype+stage)", "Elastic net mRNA", "Elastic net multi-omics"}

labels = list(data.keys())
vals = [np.asarray(data[k], dtype=float) for k in labels]

fig, ax = plt.subplots(figsize=(12.5, 6))
bp = ax.boxplot(vals, positions=range(len(labels)), widths=0.55,
                patch_artist=True, showfliers=False, whis=(0, 100))
for i, k in enumerate(labels):
    c = "#DD8452" if k in FEEDBACK else "#4C72B0"
    bp["boxes"][i].set(facecolor=c, alpha=0.30, edgecolor=c)
    bp["medians"][i].set(color=c, linewidth=2.4)
    for j in (2 * i, 2 * i + 1):
        bp["whiskers"][j].set_color(c); bp["caps"][j].set_color(c)
    ax.scatter(np.full(len(vals[i]), i), vals[i], color=c, s=45, zorder=3, edgecolor="none")

ax.axhline(0.5, ls="--", color="grey", lw=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("test C-index (5-fold CV)")
ax.set_ylim(0.25, 0.85)
ax.set_title("")
ax.legend(handles=[Patch(facecolor="#DD8452", alpha=0.5, label="post-feedback model"),
                   Patch(facecolor="#4C72B0", alpha=0.5, label="original model"),
                   Line2D([0], [0], ls="--", color="grey", label="random (0.5)")],
          loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "all_models_cindex_boxplot_extended.png", dpi=300)
plt.show()