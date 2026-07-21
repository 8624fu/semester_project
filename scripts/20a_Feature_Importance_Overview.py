# Feature-importance overview for each model and cross model comparison

# Feature-importance overview for each model and cross model comparison

# Note: raw numbers are not comparable across models.

from pathlib import Path
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = {
    "LASSO_mRNA":       "feature_importance_lasso_mrna.csv",
    "LASSO_multiomics": "feature_importance_lasso_multiomics.csv",
    "NN_mRNA":          "feature_importance_nn_mrna.csv",
    "NN_integrated":    "feature_importance_nn_integrated.csv",
}
tables = {}
for model, fname in MODELS.items():
    path = TABLES / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} - run stage 19a/19b first.")
    tables[model] = pd.read_csv(path)

print("Loaded:", {m: len(df) for m, df in tables.items()})


# Create short and readable feature label

def short_label(row):
    if row["modality"] == "RNA":
        return row["gene"]
    return f"{row['gene']} ({row['cpg']})"

for df in tables.values():
    df["label"] = df.apply(short_label, axis=1)



# Top 15 features per model

TOP_N = 15

def top_table(model):
    df = tables[model]
    cols = ["rank", "label", "modality", "importance"]
    if "selection_frequency" in df.columns:
        cols.append("selection_frequency")
    out = df.sort_values("importance", ascending=False).head(TOP_N)[cols].reset_index(drop=True)
    return out.round(4)

for model in MODELS:
    print(f"\n===== {model} — top {TOP_N} =====")
    print(top_table(model).to_string(index=False))

# Save one tidy file with every model's top-15 stacked together.
combined = pd.concat(
    [tables[m].sort_values("importance", ascending=False).head(TOP_N) for m in MODELS],
    ignore_index=True)
keep = ["model", "rank", "label", "modality", "gene", "cpg", "importance"]
if "selection_frequency" in combined.columns:
    keep.append("selection_frequency")
combined[keep].to_csv(TABLES / "feature_importance_overview_top15.csv", index=False)


# Figure 1: Importance ranking per model


fig, axes = plt.subplots(2, 2, figsize=(15, 11))

for ax, model in zip(axes.ravel(), MODELS):
    d = tables[model].sort_values("importance", ascending=False).head(12).iloc[::-1]
    colors = ["#4C72B0" if m == "RNA" else "#C44E52" for m in d["modality"]]
    ax.barh(d["label"], d["importance"], color=colors)
    metric = "mean |coef|" if model.startswith("LASSO") else "mean |SHAP|"
    ax.set_xlabel(metric); ax.set_title(model)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#4C72B0", "#C44E52")]
fig.legend(handles, ["RNA (expression)", "METH (methylation)"], loc="upper center", ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(FIGURES / "feature_importance_overview.png", dpi=300)
#plt.show()

# ------------------------------------------------------------
# Presentation colors
# ------------------------------------------------------------

BACKGROUND = "#fcfcfc"

RNA_COLOR = "#0e5f60"      # teal
METH_COLOR = "#982a50"     # pink

HEADER_COLORS = {
    "LASSO_mRNA": "#982a50",
    "LASSO_multiomics": "#982a50",
    "NN_mRNA": "#0e5f60",
    "NN_integrated": "#0e5f60",
}

TITLE_TEXT = {
    "LASSO_mRNA": "LASSO_mRNA\n(mRNA only)",
    "LASSO_multiomics": "LASSO_multiomics\n(mRNA + methylation)",
    "NN_mRNA": "NN_mRNA\n(mRNA only)",
    "NN_integrated": "NN_integrated\n(mRNA + methylation)",
}

FONTSIZE = 14
# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 7),
    facecolor=BACKGROUND
)

for i, (ax, model) in enumerate(zip(axes, MODELS)):

    # Top 12 features
    d = (
        tables[model]
        .sort_values("importance", ascending=False)
        .head(12)
        .iloc[::-1]
        .reset_index(drop=True)
    )

    bar_colors = [
        RNA_COLOR if modality == "RNA" else METH_COLOR
        for modality in d["modality"]
    ]

    # Numeric y positions
    y = np.arange(len(d))


    #make bars thicker in y-direction with height=0.62
    bars = ax.barh(
        y,
        d["importance"],
        color=bar_colors,
        height=0.9
    )

    # Set x limits before adding text
    xmax = d["importance"].max()
    ax.set_xlim(0, xmax * 1.12)

    if i == 1:
        # --------------------------------------------------------
        # Labels outside bars if the bar is too short
        # --------------------------------------------------------

        ax.set_yticks([])

        for bar, label in zip(bars, d["label"]):
            #if bar is too short label to the right of the bar
            if bar.get_width() < xmax * 0.45:
                ax.text(
                    bar.get_width() + xmax * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    va="center",
                    ha="left",
                    fontsize=FONTSIZE * 0.65,
                    fontweight="bold",
                    color="#333333",
                    clip_on=True
                )
            else:
                ax.text(
                    xmax * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    va="center",
                    ha="left",
                    fontsize=FONTSIZE * 0.65,
                    fontweight="bold",
                    color="white",
                    clip_on=True
                )

    else:
        # --------------------------------------------------------
        # Labels INSIDE bars
        # --------------------------------------------------------

        ax.set_yticks([])

        for i, (bar, label) in enumerate(zip(bars, d["label"])):
            ax.text(
                xmax * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=FONTSIZE * 0.65,
                fontweight="bold",
                color="white",
                clip_on=True
            )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    metric = "mean |coef|" if model.startswith("LASSO") else "mean |SHAP|"

    ax.set_xlabel(
        metric,
        fontsize=FONTSIZE
    )

    ax.set_ylabel("")

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.set_facecolor(BACKGROUND)

    #max number of x-ticks = 6
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(
        axis="x",
        labelsize=FONTSIZE * 0.8,
        length=3,
        colors="#333333"
    )

    ax.xaxis.label.set_color("#333333")

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.8)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

legend_handles = [
    Patch(facecolor=RNA_COLOR, edgecolor="none", label="RNA (expression)"),
    Patch(facecolor=METH_COLOR, edgecolor="none", label="Methylation")
]

legend = fig.legend(
    handles=legend_handles,
    loc="lower right",
    bbox_to_anchor=(0.99, -0.05),
    ncol=2,
    frameon=True,
    fancybox=False,
    fontsize=FONTSIZE,
    facecolor=BACKGROUND,
    edgecolor="#d8d8d8"
)

for text in legend.get_texts():
    text.set_color("#333333")

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

#reduce space between subplots


# plt.subplots_adjust(
#     left=0.04,
#     right=0.99,
#     top=0.83,
#     bottom=0.16,
#     wspace=0.42
# )

plt.tight_layout()
plt.savefig(
    FIGURES / "feature_importance_overview_presentation.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=BACKGROUND
)

#plt.show()


# Figure 2: LASSO selection frequency

lasso_models = [m for m in MODELS if "selection_frequency" in tables[m].columns]

fig, axes = plt.subplots(1, len(lasso_models), figsize=(7.5 * len(lasso_models), 6))
if len(lasso_models) == 1:
    axes = [axes]

for ax, model in zip(axes, lasso_models):
    d = tables[model].sort_values("selection_frequency", ascending=False).head(12).iloc[::-1]
    colors = ["#4C72B0" if m == "RNA" else "#C44E52" for m in d["modality"]]
    ax.barh(d["label"], d["selection_frequency"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of 25 fits selected"); ax.set_title(model)

plt.tight_layout()
plt.savefig(FIGURES / "feature_importance_lasso_selection_frequency.png", dpi=300)
#plt.show()



# Cross model consistency counter.


K = 15

# each model's top-K genes (collapse CpGs to gene via the gene's best feature)
model_topk = {}
for model, df in tables.items():
    gene_best = df.groupby("gene")["importance"].max().sort_values(ascending=False)
    model_topk[model] = list(gene_best.head(K).index)

# for every gene: how many models rank it in their top K, and which ones
from collections import defaultdict
in_models = defaultdict(list)
for model, genes in model_topk.items():
    for g in genes:
        in_models[g].append(model)

rec = pd.DataFrame([{"gene": g, "n_models": len(ms), "models": ", ".join(ms)}
                    for g, ms in in_models.items()])
rec = rec[rec["n_models"] >= 2].sort_values(["n_models", "gene"]).reset_index(drop=True)
rec.to_csv(TABLES / "feature_importance_topK_recurrence.csv", index=False)
print(f"Genes in the top {K} of >= 2 models:")
print(rec.sort_values("n_models", ascending=False).to_string(index=False))

# Stacked bar: one segment per model that has the gene in its top K, coloured by model.
MODEL_COLORS = {"LASSO_mRNA": "#B22222", "LASSO_multiomics": "#E8776B",
                "NN_mRNA": "#2E7D32", "NN_integrated": "#81C784"}
d = rec.sort_values("n_models")
genes = d["gene"].tolist()
present = pd.DataFrame(0, index=genes, columns=list(MODELS))
for _, row in d.iterrows():
    for m in row["models"].split(", "):
        present.loc[row["gene"], m] = 1

plt.figure(figsize=(8, max(4, 0.4 * len(genes))))
left = np.zeros(len(genes))
for model in MODELS:
    plt.barh(genes, present[model].values, left=left,
             color=MODEL_COLORS[model], label=model)
    left += present[model].values
plt.xlim(0, 4); plt.xticks([0, 1, 2, 3, 4])
plt.xlabel(f"number of models with this gene in their top {K} (coloured by model)")
plt.title("Cross-model consistency — which models agree")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "feature_importance_consistency_counter.png", dpi=300)
#plt.show()

K = 15
models = list(tables.keys())
FONT_SIZE = 15
TEXT_COLOR = "#222222"
GENE_COLORS = {
    "RNA": "#0e5f60ff",
    "METH": "#982a50ff",
}



# =========================================================
# 1. Identify each model's top-K genes
# =========================================================

model_topk = {}
model_topk_labels = {}
model_topk_types = {}

for model, df in tables.items():

    # Find the highest-importance feature associated with each gene.
    best_idx = df.groupby("gene")["importance"].idxmax()

    gene_best = (
        df.loc[
            best_idx,
            ["gene", "label", "modality", "importance"],
        ]
        .sort_values("importance", ascending=False)
        .head(K)
        .copy()
    )

    model_topk[model] = gene_best["gene"].tolist()

    # Gene -> display label
    model_topk_labels[model] = dict(
        zip(
            gene_best["gene"],
            gene_best["label"],
        )
    )

    # Gene -> RNA or METH
    model_topk_types[model] = dict(
        zip(
            gene_best["gene"],
            gene_best["modality"],
        )
    )


# =========================================================
# 3. Construct the binary gene × model matrix
# =========================================================

genes = rec["gene"].tolist()

present = pd.DataFrame(
    0,
    index=genes,
    columns=models,
    dtype=int,
)

for model, genes_selected in model_topk.items():
    genes_in_matrix = present.index.intersection(genes_selected)
    present.loc[genes_in_matrix, model] = 1


# Sort by consensus count, then alphabetically.
row_order = (
    present.assign(
        n_models=present.sum(axis=1),
        gene_name=present.index,
    )
    .sort_values(
        ["n_models", "gene_name"],
        ascending=[False, True],
    )
    .index
)

present = present.loc[row_order]
genes = present.index.tolist()


# =========================================================
# 4. Create gene-to-display-label mapping
# =========================================================

gene_to_label = {}

for model in models:
    for gene, label in model_topk_labels[model].items():
        if gene not in gene_to_label:
            gene_to_label[gene] = label


y_labels = [
    gene_to_label.get(gene, gene).split(" (")[0]
    for gene in genes
]


# =========================================================
# 5. Plot the binary heatmap
# =========================================================

fig_height = max(4, 0.45 * len(genes))

fig, ax = plt.subplots(
    figsize=(11, fig_height)
)


for row_idx, gene in enumerate(genes):
    for col_idx, model in enumerate(models):

        selected = present.loc[gene, model] == 1

        if selected:
            facecolor = GENE_COLORS[model_topk_types[model][gene]]
        else:
            facecolor = "white"

        cell = Rectangle(
            (col_idx, row_idx),
            width=1,
            height=1,
            facecolor=facecolor,
            edgecolor="#D0D0D0",
            linewidth=0.8,
        )

        ax.add_patch(cell)


# =========================================================
# 6. Format axes
# =========================================================

ax.set_xlim(0, len(models))
ax.set_ylim(0, len(genes))

ax.set_xticks(
    np.arange(len(models)) + 0.5
)

ax.set_xticklabels(
    models,
    rotation=30,
    ha="right",
    size=FONT_SIZE*0.8,
    color = TEXT_COLOR
)

ax.set_yticks(
    np.arange(len(genes)) + 0.5
)

ax.set_yticklabels(
    y_labels,
    size=FONT_SIZE*0.8,
    color = TEXT_COLOR
)

ax.invert_yaxis()

# ax.set_xlabel("Model", fontsize=FONT_SIZE, color=TEXT_COLOR)
ax.set_ylabel("Gene label", fontsize=FONT_SIZE, color=TEXT_COLOR)

ax.set_title(
    "Cross-model feature consistency\n"
    f"Genes appearing in the top {K} of at least two models",
    fontsize=FONT_SIZE,
    color = TEXT_COLOR
)

# =========================================================
# 8. Add feature-type legend
# =========================================================

legend_handles = [
    Patch(
        facecolor=GENE_COLORS["METH"],
        edgecolor="#D0D0D0",
        label="Methylation",
    ),
    Patch(
        facecolor=GENE_COLORS["RNA"],
        edgecolor="#D0D0D0",
        label="RNA",
    )
]

ax.legend(
    handles=legend_handles,
    bbox_to_anchor=(1.3, 0.5),
    loc="center right",
    frameon=False,
    fontsize=FONT_SIZE*0.8,
    labelcolor = TEXT_COLOR
)


for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(
    axis="both",
    length=0,
)


# =========================================================
# 9. Save and display
# =========================================================

plt.tight_layout()

plt.savefig(
    FIGURES / "feature_importance_consistency_binary_heatmap_presentation.png",
    dpi=300,
    bbox_inches="tight",
)

#plt.show()


## Summary

# Outputs:

# - feature_importance_overview_top15.csv 
# - feature_importance_overview.png
# - feature_importance_lasso_selection_frequency.png 
# - feature_importance_topK_recurrence.csv` / `feature_importance_consistency_counter.png