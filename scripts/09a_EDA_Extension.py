#!/usr/bin/env python
# coding: utf-8


# 09a — EDA Extension

# Extended exploratory data analysis checking:
# - batch effect
# - methylation distribution after transformation (beta -> m-values)
# - subtype signal: PCA colored by subtype

# Note: Analyses use full-cohort KNN-imputed dataset (`meth_pam50_knn_imputed.csv`)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]

Path(PROJECT_ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
Path(PROJECT_ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)


rna      = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "rna_pam50.csv").set_index("patient")
meth     = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "meth_pam50_knn_imputed.csv", index_col=0)
meth_raw = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "meth_pam50.csv").set_index("patient")
labels   = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "labels_luminal_brca.csv").set_index("patient")
cpg_gene = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cpg_gene_map.csv")
surv     = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "survival_luminal_clean.csv").set_index("patient")

patients = rna.index.intersection(meth.index).intersection(labels.index)
rna, meth, labels = rna.loc[patients], meth.loc[patients], labels.loc[patients]
subtype  = labels["subtype"]
meth_clean = meth.dropna(axis=1, how="all")
meth_scaled = StandardScaler().fit_transform(meth_clean)

COLORS = {"LumA": "#0e5f60", "LumB": "#9c224d"}
print(f"Patients: {len(patients)} | LumA: {(subtype=='LumA').sum()} | LumB: {(subtype=='LumB').sum()}")
print(f"RNA genes: {rna.shape[1]} | CpGs (imputed): {meth_clean.shape[1]}")


# Methylation PCA colored by TSS
tss = pd.Series([pid.split("-")[1] for pid in meth_clean.index],
                index=meth_clean.index, name="TSS")
meth_scaled_batch = StandardScaler().fit_transform(meth_clean)
pca_batch = PCA(n_components=10, random_state=42)
pcs_batch = pca_batch.fit_transform(meth_scaled_batch)

fig, ax = plt.subplots(figsize=(10, 7))
for site in tss.unique():
    mask = (tss == site).values
    ax.scatter(pcs_batch[mask, 0], pcs_batch[mask, 1], label=site, s=15, alpha=0.6)
ax.set_xlabel(f"PC1 ({pca_batch.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_batch.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Methylation PCA colored by tissue source site (TSS)")
ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "figures" / "meth_pca_batch_check.png", dpi=300)
#plt.show()
tss_enc = LabelEncoder().fit_transform(tss)
meth_per_pc = [LinearRegression().fit(tss_enc.reshape(-1,1), pcs_batch[:,i].reshape(-1,1))
                  .score(tss_enc.reshape(-1,1), pcs_batch[:,i].reshape(-1,1)) for i in range(10)]

# RNA PCA colored by TSS
tss_rna = pd.Series([pid.split("-")[1] for pid in rna.index],
                     index=rna.index, name="TSS")
rna_scaled_batch = StandardScaler().fit_transform(rna)
pca_rna_b = PCA(n_components=10, random_state=42)
pcs_rna_b = pca_rna_b.fit_transform(rna_scaled_batch)

fig, ax = plt.subplots(figsize=(10, 7))
for site in tss_rna.unique():
    mask = (tss_rna == site).values
    ax.scatter(pcs_rna_b[mask, 0], pcs_rna_b[mask, 1], label=site, s=15, alpha=0.6)
ax.set_xlabel(f"PC1 ({pca_rna_b.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_rna_b.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("RNA PCA colored by tissue source site (TSS)")
ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "figures" / "rna_pca_batch_check.png", dpi=300)
#plt.show()
tss_rna_enc = LabelEncoder().fit_transform(tss_rna)
rna_per_pc = [LinearRegression().fit(tss_rna_enc.reshape(-1,1), pcs_rna_b[:,i].reshape(-1,1))
                  .score(tss_rna_enc.reshape(-1,1), pcs_rna_b[:,i].reshape(-1,1)) for i in range(10)]


tab = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(10)],
    "Meth var (%)":   [round(v*100, 1) for v in pca_batch.explained_variance_ratio_],
    "Meth batch R²":  [round(v, 4) for v in meth_per_pc],
    "RNA var (%)":    [round(v*100, 1) for v in pca_rna_b.explained_variance_ratio_],
    "RNA batch R²":   [round(v, 4) for v in rna_per_pc],
})
mean_row = pd.DataFrame([["Mean", "", round(np.mean(meth_per_pc), 4),
                           "", round(np.mean(rna_per_pc), 4)]],
                         columns=tab.columns)
tab = pd.concat([tab, mean_row], ignore_index=True)
print(tab.to_string(index=False))

# Both methylation (mean R²=0.007) and RNA (mean R²=0.010) show negligible batch effects across PC1–10. The highest single value is RNA PC6 (R²=0.036), which itself explains only 3.5% of total RNA variance.

# Since there still is a small cluster visible in methylation data, it is further investigated:

# extract cluster from PCA coordinates
pca_df_meth = pd.DataFrame(pcs_batch[:, :2],
                             index=meth_clean.index,
                             columns=["PC1", "PC2"])
cluster_patients = pca_df_meth[
    (pca_df_meth["PC1"] > 15) & (pca_df_meth["PC2"] < -7)
].index.tolist()
rest_patients = [p for p in meth_clean.index if p not in cluster_patients]

print(f"Patients: {len(cluster_patients)}")
print(cluster_patients)

# TSS and subtype breakdown of cluster patients
cluster_tss = pd.Series([p.split("-")[1] for p in cluster_patients],
                         index=cluster_patients, name="TSS")
cluster_subtype = labels.loc[cluster_patients, "subtype"]

crosstab_cluster = pd.crosstab(cluster_tss, cluster_subtype)
crosstab_cluster["Total"] = crosstab_cluster.sum(axis=1)
print("Composition (TSS x subtype):")
print(crosstab_cluster.to_string())
#most patients from E9, but also three other sites -> what connects them?

# Investigate whether cluster driven by imputation:
# <- then cluster patients should have higher missingness than rest of cohort

# missingness comparison: cluster vs rest
cluster_missing = meth_raw.loc[cluster_patients].isna().mean(axis=1)
rest_missing    = meth_raw.loc[rest_patients].isna().mean(axis=1)
print(f"\nMean missingness — cluster: {cluster_missing.mean():.3f} | rest: {rest_missing.mean():.3f}")
print(f"Max missingness  — cluster: {cluster_missing.max():.3f} | rest: {rest_missing.max():.3f}")

rest_missing_s    = pd.Series(rest_missing)
cluster_missing_s = pd.Series(cluster_missing)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.hist(rest_missing_s, bins=30, alpha=0.6, color="#7F77DD",
        edgecolor="#5c54c4", linewidth=1.2,
        label=f"Rest (n={len(rest_missing_s)})", density=True)

plotted_line = False
for patient, val in cluster_missing_s.items():
    ax.axvline(val, color="#D85A30", linewidth=1.8, linestyle="--",
               label=f"Cluster patients (n={len(cluster_missing_s)})" if not plotted_line else None)
    plotted_line = True

ax.set_xlabel("Fraction of missing CpGs per patient")
ax.set_ylabel("Density")
ax.set_title("Missingness before imputation: cluster vs rest")
ax.legend(frameon=True, facecolor="white", edgecolor="none")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
#plt.savefig(PROJECT_ROOT / "results" / "figures" / "cluster_missingness_comparison.png", dpi=300, bbox_inches="tight")
#plt.show()
# -> cluster mean missingness with .112 a bit higher than rest (.097), but max missingness nearly identical. Imputation doesn't look like the driver of the cluster.



# Investigate whether unusual methylation levels could drive the cluster:
# <- then cluster patients should have higher mean beta values than rest of cohort

# mean beta: cluster vs rest
cluster_mean_beta = meth_clean.loc[cluster_patients].mean(axis=1)
rest_mean_beta    = meth_clean.loc[rest_patients].mean(axis=1)
print("Mean beta — all patients:")
print(f"  Cluster: {cluster_mean_beta.mean():.4f} (n={len(cluster_patients)}) "
      f"| Rest: {rest_mean_beta.mean():.4f} (n={len(rest_patients)})")
#not a difference that would explain the cluster -> differentiate by subtype

# mean beta by subtype: cluster vs rest
print("\nMean beta by subtype — cluster vs rest:")
for st in ["LumA", "LumB"]:
    cl = [p for p in cluster_patients if labels.loc[p, "subtype"] == st]
    rs = [p for p in rest_patients    if labels.loc[p, "subtype"] == st]
    if cl:
        print(f"  {st} — cluster: {meth_clean.loc[cl].mean().mean():.4f} (n={len(cl)}) "
              f"| rest: {meth_clean.loc[rs].mean().mean():.4f} (n={len(rs)})")

# plots for LumA and LumB
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, st in enumerate(["LumA", "LumB"]):
    ax = axes[i]
    st_rest_p    = [p for p in rest_patients    if labels.loc[p, "subtype"] == st]
    st_cluster_p = [p for p in cluster_patients if labels.loc[p, "subtype"] == st]

    if st_rest_p:
        ax.hist(rest_mean_beta.loc[st_rest_p], bins=20, alpha=0.4,
                color=COLORS[st], edgecolor=COLORS[st], linewidth=1.2,
                label=f"Rest {st} (n={len(st_rest_p)})", density=True)

    plotted_line = False
    for patient in st_cluster_p:
        ax.axvline(cluster_mean_beta[patient], color=COLORS[st],
                   linewidth=1.8, linestyle="--",
                   label=f"Cluster {st} (n={len(st_cluster_p)})" if not plotted_line else None)
        plotted_line = True

    ax.set_xlabel("Mean beta value (post-imputation)")
    ax.set_ylabel("Density" if i == 0 else "")
    ax.set_title(f"{st} Subtype")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Mean beta distribution: cluster vs rest", fontsize=14, y=1.02)
plt.tight_layout()
#plt.savefig(PROJECT_ROOT / "results" / "figures" / "cluster_mean_beta_comparison.png", dpi=300, bbox_inches="tight")
#plt.show()
#within LumA subtype, cluster beta values seem to be normally distributed; LumB luster patients show a more hypermethylated 
#distribution, with mean beta 0.187 vs 0.166 in LumB rest.


# --> Final conclusion:
# The cluster (n=16) spans four tissue source sites, with E9 (Asterand) contributing 11 patients across both subtypes. Cluster patients show 
# similar missingness to the rest (mean 11.2% vs 9.7%, max nearly identical), suggesting imputation is not the driver. The cluster more plausibly reflects 
# a biologically hypermethylated subgroup, particularly among LumB patients (mean beta 0.187 vs 0.166 in LumB rest). These 16 patients (2.8% of the 
# cohort) are retained in all downstream analyses.



def beta_to_m(B):
    B = np.clip(B, 1e-4, 1 - 1e-4)
    return np.log2(B / (1 - B))

BG = "#fcfcfc"
TEXT_SIZE = 19
COL_BETA = "#9c224d"
COL_M    = "#0e5f60"

all_beta = meth_clean.values.ravel()
all_beta = all_beta[~np.isnan(all_beta)]
all_m = beta_to_m(all_beta)

fig, axes = plt.subplots(2, 1, figsize=(8, 9), facecolor=BG)
fig.patch.set_facecolor(BG)

for row_idx, (vals, color, xlabel) in enumerate([
    (all_beta, COL_BETA, "Beta value"),
    (all_m,    COL_M,    "M-value")
]):
    ax = axes[row_idx]
    ax.set_facecolor(BG)
    ax.hist(vals, bins=60, color=color, edgecolor="none", density=True, alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=TEXT_SIZE * 0.75)
    ax.set_ylabel("Density", fontsize=TEXT_SIZE * 0.75)
    ax.tick_params(axis="both", labelsize=TEXT_SIZE * 0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1)

fig.suptitle(
    "PAM50 Promoter Methylation — Beta and M-value Distributions",
    fontsize=TEXT_SIZE,
    color="#9c224d",
    fontweight="bold",
    y=1.01
)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "figures" / "beta_vs_mvalue_distribution.png", dpi=300,
            facecolor=BG, bbox_inches="tight")
#plt.show()

# Beta values are strongly right-skewed, most PAM50 promoter CpGs are constitutively unmethylated.
# The M-value transformation produces a more symmetric distribution and reduces heteroscedasticity,
# making it better suited for modeling than raw beta values.



# RNA 

scaler = StandardScaler()
rna_scaled = scaler.fit_transform(rna)
pca_rna = PCA(n_components=5)
rna_pcs = pca_rna.fit_transform(rna_scaled)
var_rna = pca_rna.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (px, py) in zip(axes, [(0,1),(1,2)]):
    for st in ["LumA", "LumB"]:
        mask = (subtype == st).values
        ax.scatter(rna_pcs[mask, px], rna_pcs[mask, py], c=COLORS[st], label=st, s=18, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"PC{px+1} ({var_rna[px]:.1f}%)")
    ax.set_ylabel(f"PC{py+1} ({var_rna[py]:.1f}%)")
    ax.set_title(f"RNA PCA: PC{px+1} vs PC{py+1}")
    ax.legend(title="Subtype"); ax.spines[["top","right"]].set_visible(False)
fig.suptitle("PCA of PAM50 RNA Expression (50 genes)", fontsize=12)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "figures" / "rna_pca_by_subtype.png", dpi=300)
#plt.show()


# Methylation 

pca_meth = PCA(n_components=5, random_state=42)
meth_pcs = pca_meth.fit_transform(meth_scaled)
var_meth = pca_meth.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (px, py) in zip(axes, [(0,1),(1,2)]):
    for st in ["LumA","LumB"]:
        mask = (subtype==st).values
        ax.scatter(meth_pcs[mask,px], meth_pcs[mask,py], c=COLORS[st], label=st, s=18, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"PC{px+1} ({var_meth[px]:.1f}%)")
    ax.set_ylabel(f"PC{py+1} ({var_meth[py]:.1f}%)")
    ax.set_title(f"Methylation PCA: PC{px+1} vs PC{py+1}")
    ax.legend(title="Subtype"); ax.spines[["top","right"]].set_visible(False)
fig.suptitle("PCA of PAM50 Promoter Methylation (post-imputation)", fontsize=12)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "figures" / "meth_pca_by_subtype.png", dpi=300)
#plt.show()


# RNA PC1 explains 39.8% of RNA variance and shows partial LumB enrichment on one side, but also shows substantial overlap between 
# subtypes. This is expected: LumA and LumB are closely related, differing mainly in proliferation rate.
# The RNA baseline model therefore has a real but moderate subtype signal to work with.

# Methylation PCA shows PC1 explains only 12.5% of variance and LumA/LumB are largely intermixed -> Methylation does not separate 
# the subtypes in linear feature space
# -> This suggests the two layers carry partially different information; methylation is not simply redundant with expression, 
# which provides one motivation for their integration.


