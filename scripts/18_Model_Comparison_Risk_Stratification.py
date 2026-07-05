#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from pathlib import Path

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("lifelines not installed. pip install lifelines")

Path("../results/figures").mkdir(parents=True, exist_ok=True)
Path("../results/tables").mkdir(parents=True, exist_ok=True)

COLORS = {
    "mRNA-only LASSO":   "#7F77DD",
    "Multi-omics LASSO": "#1D9E75",
    "mRNA-only NN":      "#D85A30",
    "Multi-omics NN":    "#BA7517",
}
ORDER = ["mRNA-only LASSO", "Multi-omics LASSO", "mRNA-only NN", "Multi-omics NN"]


# In[2]:

# load all four model result files
lasso_mrna  = pd.read_csv("../results/tables/lasso_cox_cv_results.csv")
lasso_multi = pd.read_csv("../results/tables/lasso_cox_multiomics_cv_results.csv")
nn_mrna     = pd.read_csv("../results/tables/nn_mRNA_only_best_model_folds.csv")
nn_multi    = pd.read_csv("../results/tables/nn_integrated_best_model_folds.csv")

surv = pd.read_csv("../data/processed/survival_luminal_clean.csv").set_index("patient")

lasso_mrna["model"]  = "mRNA-only LASSO"
lasso_multi["model"] = "Multi-omics LASSO"
nn_mrna["model"]     = "mRNA-only NN"
nn_multi["model"]    = "Multi-omics NN"

combined = pd.concat([lasso_mrna, lasso_multi, nn_mrna, nn_multi], ignore_index=True)
print("All four models loaded.")
print(combined.groupby("model")["test_c_index"].agg(["mean","std"]).round(3))


# In[3]:

# performance summary table
summary = (
    combined.groupby("model")["test_c_index"]
    .agg(mean="mean", std="std", min="min", max="max")
    .round(3)
    .reset_index()
)
summary["model"] = pd.Categorical(summary["model"], categories=ORDER, ordered=True)
summary = summary.sort_values("model").reset_index(drop=True)
summary.to_csv("../results/tables/all_models_performance_summary.csv", index=False)
print(summary.to_string(index=False))


# In[4]:

# visualization of mean C-index +/- SD
fig, ax = plt.subplots(figsize=(7, 5))
for _, row in summary.iterrows():
    ax.errorbar(row["model"], row["mean"], yerr=row["std"],
                fmt="o", capsize=6, markersize=9, color=COLORS[row["model"]])
    ax.text(row["model"], row["mean"] + row["std"] + 0.015,
            f'{row["mean"]:.3f}', ha="center", fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_title("Mean C-index ± SD (5-fold CV)")
ax.set_ylabel("C-index")
ax.set_ylim(0.25, 0.85)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/cindex_all_models_mean_sd.png", dpi=300)
plt.show()


# In[5]:

# C-index boxplot
model_order = [m for m in ORDER if m in combined["model"].unique()]
palette = {m: COLORS[m] for m in model_order}

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=combined, x="model", y="test_c_index", order=model_order,
            palette=palette, width=0.45, ax=ax)
sns.stripplot(data=combined, x="model", y="test_c_index", order=model_order,
              color="black", alpha=0.7, size=7, ax=ax)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
ax.set_title("C-index comparison — all models (5-fold CV)")
ax.set_xlabel("")
ax.set_ylabel("C-index")
ax.set_ylim(0.25, 0.85)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/cindex_all_models_boxplot.png", dpi=300)
plt.show()


# In[6]:

# per-fold C-index bar chart
fig, ax = plt.subplots(figsize=(9, 5))
n_models = len(model_order)
width = 0.18
x = np.arange(1, 6)

for i, model in enumerate(model_order):
    subset = combined[combined["model"] == model].sort_values("fold")
    offset = (i - (n_models - 1) / 2) * width
    ax.bar(x + offset, subset["test_c_index"], width,
           label=model, color=COLORS[model], alpha=0.85)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {f}" for f in range(1, 6)])
ax.set_ylabel("C-index")
ax.set_title("Per-fold C-index — all models")
ax.set_ylim(0.25, 0.85)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/cindex_per_fold_all_models.png", dpi=300)
plt.show()


# In[7]:

# feature selection — LASSO models only
lasso_only = combined[combined["model"].isin(["mRNA-only LASSO", "Multi-omics LASSO"])]

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=lasso_only, x="model", y="n_features_selected",
            palette={"mRNA-only LASSO": COLORS["mRNA-only LASSO"],
                     "Multi-omics LASSO": COLORS["Multi-omics LASSO"]},
            width=0.4, ax=ax)
sns.stripplot(data=lasso_only, x="model", y="n_features_selected",
              color="black", alpha=0.7, size=7, ax=ax)
ax.set_title("Features selected by LASSO per fold")
ax.set_xlabel("")
ax.set_ylabel("Features selected")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/feature_selection_stability.png", dpi=300)
plt.show()


# In[8]:

# wilcoxon signed-rank tests between key pairs
pairs = [
    ("mRNA-only LASSO",   "Multi-omics LASSO",  "Does methylation help in LASSO?"),
    ("mRNA-only NN",      "Multi-omics NN",      "Does methylation help in NN?"),
    ("mRNA-only LASSO",   "mRNA-only NN",        "Does NN beat LASSO on RNA only?"),
    ("Multi-omics LASSO", "Multi-omics NN",      "Does NN beat LASSO with both omics?"),
]

wilcoxon_rows = []
for m1, m2, question in pairs:
    s1 = combined.loc[combined["model"] == m1, "test_c_index"].values
    s2 = combined.loc[combined["model"] == m2, "test_c_index"].values
    stat, pval = wilcoxon(s1, s2)
    sig = "yes" if pval < 0.05 else "no"
    wilcoxon_rows.append({"comparison": question, "model_1": m1, "model_2": m2,
                          "statistic": round(stat, 3), "p_value": round(pval, 4),
                          "significant": sig})
    print(f"{question}: p={pval:.4f} ({'significant' if pval < 0.05 else 'not significant'})")

pd.DataFrame(wilcoxon_rows).to_csv(
    "../results/tables/wilcoxon_model_comparisons.csv", index=False)


# In[9]:

# KM curves from predicted risk scores
km_files = {
    "mRNA-only LASSO":   "../results/tables/lasso_cox_mrna_risk_scores.csv",
    "Multi-omics LASSO": "../results/tables/lasso_cox_multiomics_risk_scores.csv",
    "mRNA-only NN":      "../results/tables/nn_mrna_only_risk_scores.csv",
    "Multi-omics NN":    "../results/tables/nn_integrated_risk_scores.csv",
}

km_results = []
for model_name, fpath in km_files.items():
    if not Path(fpath).exists():
        print(f"Skipping {model_name} — {fpath} not found")
        continue

    scores = pd.read_csv(fpath).set_index("patient")
    common = scores.index.intersection(surv.index)
    d = surv.loc[common].copy()
    d["risk_score"] = scores.loc[common, "risk_score"]
    d["time_years"] = d["time"] / 365.25
    d["risk_group"] = np.where(d["risk_score"] >= d["risk_score"].median(),
                               "High risk", "Low risk")

    lr = logrank_test(
        d.loc[d["risk_group"] == "Low risk",  "time_years"],
        d.loc[d["risk_group"] == "High risk", "time_years"],
        event_observed_A=d.loc[d["risk_group"] == "Low risk",  "event"],
        event_observed_B=d.loc[d["risk_group"] == "High risk", "event"]
    )

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))
    for group, color in [("Low risk", "#1D9E75"), ("High risk", "#D85A30")]:
        sub = d[d["risk_group"] == group]
        kmf.fit(sub["time_years"], sub["event"], label=f"{group} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Overall survival (years)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"KM — predicted risk groups: {model_name}\nlog-rank p={lr.p_value:.4g}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(f"../results/figures/km_risk_groups_{fname}.png", dpi=300)
    plt.show()

    km_results.append({"model": model_name, "logrank_p": round(lr.p_value, 4),
                       "n_high": (d["risk_group"] == "High risk").sum(),
                       "n_low":  (d["risk_group"] == "Low risk").sum()})

if km_results:
    km_df = pd.DataFrame(km_results)
    km_df.to_csv("../results/tables/km_risk_group_logrank.csv", index=False)


# In[10]:

# final summary table
km_df = pd.read_csv("../results/tables/km_risk_group_logrank.csv")
report_table = summary.copy()
report_table.columns = ["Model", "Mean C-index", "SD", "Min", "Max"]
report_table = report_table.merge(
    km_df[["model", "logrank_p"]].rename(
        columns={"model": "Model", "logrank_p": "KM log-rank p"}),
    on="Model", how="left"
)
report_table.to_csv("../results/tables/final_model_comparison_table.csv", index=False)
print(report_table.to_string(index=False))


