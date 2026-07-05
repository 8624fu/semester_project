01_Initial_Data_Download.R:

Purpose: Download raw multi-omics data from TCGA-BRCA

1. Installs and loads required Bioconductor packages (TCGAbiolinks, SummarizedExperiment, sesame)
Queries and downloads:
- DNA methylation (Illumina Human Methylation 450K beta values)
- RNA-seq gene expression (STAR - Counts workflow)
- Clinical metadata from TCGA-BRCA

2. Converts GDC query results into R SummarizedExperiment objects

3. Saves raw datasets as .RDS files for reproducible reuse:
- meth_tcga_brca.rds
- rna_tcga_brca.rds
- clinical_brca.rds

--------------------------------------------------------------------------------

02_Merge_Data.R:

Purpose: Preprocess and harmonize datasets for Luminal A / Luminal B analysis

Loads previously saved raw TCGA datasets
Retrieves PAM50 molecular subtype classification using TCGAquery_subtype()

Filters cohort to include only:
- Luminal A (LumA)
- Luminal B (LumB)

Ensures multi-omics availability by intersecting:
- Methylation samples
- RNA-seq samples
- Clinical samples
- Luminal subtype patients
Results in a harmonized cohort of patients with complete data across all modalities

Subsets all datasets to shared patients

Adds PAM50 subtype labels to clinical metadata

Synchronizes sample order across all datasets to ensure correct alignment

Extracts final analysis matrices:
- beta (CpG × samples methylation matrix)
- rna_mat (gene × samples expression matrix)
- y (LumA vs LumB labels)

Packages all objects into a single multi-omics list

Saves final processed dataset as:
- multiomics_luminal_brca.rds

--------------------------------------------------------------------------------

03_Check_Missings.R:

Load processed multi-omics dataset for downstream analysis

Extracts:
- DNA methylation matrix (beta)
- RNA expression matrix (rna_mat)
- Clinical metadata (clinical_common)
- PAM50 labels (y)

Provides ready-to-use objects for downstream analyses.

- ToDo: Check Missing Values / Exploratory Analsis.

--------------------------------------------------------------------------------

04_Feature_Selection.R:

Purpose: Restrict the harmonized cohort to PAM50 features

Loads the harmonized dataset (multiomics_luminal_brca.rds) whose beta, rna,
clinical and y are already aligned to the same 563 patients.

RNA:
- Maps Ensembl IDs to gene symbols and keeps the 50 PAM50 genes
- Normalizes raw STAR counts with DESeq2 size factors (estimated on all genes),
  then log2-transforms the normalized PAM50 counts

Methylation:
- Loads the Illumina 450K annotation
- Keeps CpGs whose PAM50 gene is paired with a promoter region
  (TSS1500, TSS200 or 1stExon), correctly handling the ';'-separated,
  positionally paired gene/region annotation
- Includes legacy symbol aliases (CDCA1, KNTC2, ORC6L) so promoter CpGs of
  NUF2, NDC80 and ORC6 are not missed in the hg19 annotation

Alignment and checks:
- Reduces RNA and methylation column names to the 12-char patient ID and
  verifies identical sample order across layers
- Reports methylation missingness per sample and per CpG (KNN imputation follows)

Output:
- pam50_features_brca.rds (list: rna, meth, clinical, y)

--------------------------------------------------------------------------------

05_export_for_python.R:

Purpose: Export the PAM50 feature dataset from R to Python-friendly CSV files

Loads:

* pam50_features_brca.rds

Verifies that RNA, methylation, clinical metadata, and subtype labels are aligned
to the same patient order.

Transposes the RNA and methylation matrices so that:

* rows = patients
* columns = molecular features

Exports:

* rna_pam50.csv
* meth_pam50.csv
* clinical_luminal_brca.csv
* labels_luminal_brca.csv

These files are saved in data/processed/ for use in Python-based exploratory
analysis, cross-validation, imputation, and survival modeling.

--------------------------------------------------------------------------------

05b_export_cpg_gene_map.R:

Purpose: Export a CpG -> PAM50 gene mapping for the Python correlation analysis

Loads:

* pam50_features_brca.rds (for the list of selected promoter CpGs)
* Illumina 450K annotation

Reproduces the stage-04 probe-selection logic restricted to the selected promoter
CpGs, pairing each CpG with the PAM50 gene(s) whose promoter region it covers
(TSS1500, TSS200, 1stExon) and converting legacy symbols (CDCA1, KNTC2, ORC6L)
to current symbols (NUF2, NDC80, ORC6) so methylation and RNA gene names match.

Output:

* cpg_gene_map.csv (one row per CpG-gene pair)

--------------------------------------------------------------------------------

06_week1_eda.ipynb / 06_week1_eda.py:

Purpose: Perform initial exploratory data analysis of the exported PAM50 cohort

Loads the Python-ready RNA, methylation, clinical, and subtype-label files.

Checks:

* patient alignment across all exported datasets
* number of patients and molecular features
* Luminal A and Luminal B subtype distribution
* RNA and methylation missingness
* availability and structure of overall survival variables

Generates:

* subtype distribution plot
* methylation missingness per patient plot
* methylation missingness per CpG plot
* survival event proportion by subtype plot
* Week 1 cohort summary table

Creates and saves a cleaned survival outcome table containing:

* patient ID
* PAM50 subtype
* survival event indicator
* overall survival time

--------------------------------------------------------------------------------

06b_EDA_Extension.ipynb / 06b_EDA_Extension.py:

Purpose: Extended exploratory data analysis covering RNA subtype signal,
dimensionality reduction across both omics layers, and the multi-gene
methylation-expression relationship.

Loads:

* rna_pam50.csv
* meth_pam50_knn_imputed.csv (post-imputation, for plotting only)
* labels_luminal_brca.csv
* cpg_gene_map.csv

Generates:

* Violin plots of log2-normalized expression for 8 key PAM50 genes split by
  LumA and LumB with Mann-Whitney U significance labels
* Mann-Whitney U test with Benjamini-Hochberg correction for all 50 PAM50 genes
* PCA of the 50 PAM50 RNA features colored by PAM50 subtype
* PCA of the imputed promoter methylation matrix colored by PAM50 subtype
* UMAP of the 50 PAM50 RNA features colored by PAM50 subtype
* UMAP of the imputed promoter methylation matrix colored by PAM50 subtype
* Scatter panel of mean promoter beta vs log2 expression for 8 selected PAM50
  genes with per-gene Spearman rho
* Methylation violin plots for the same 8 genes split by subtype

Saves:

* rna_violin_by_subtype.png
* rna_pca_by_subtype.png
* meth_pca_by_subtype.png
* rna_umap_by_subtype.png
* meth_umap_by_subtype.png
* meth_vs_rna_scatter_panel.png
* meth_violin_by_subtype.png
* rna_pca_loadings.csv
* rna_subtype_significance_all_genes.csv

Note: uses the full-cohort imputed methylation file for visualizations only.
For cross-validated survival modeling, fold-safe imputation must be used instead.

Requires umap-learn (pip install umap-learn).
Run after scripts 06, 07, 08 and 05b.

--------------------------------------------------------------------------------

07_create_cv_folds.ipynb / 07_create_cv_folds.py:

Purpose: Create reproducible stratified cross-validation folds for survival modeling

Loads the cleaned survival outcome table created during Week 1 EDA.

Creates a combined stratification label based on:

* PAM50 subtype (LumA or LumB)
* survival event status (event or censored)

Uses 5-fold stratified cross-validation to preserve the approximate distribution
of Luminal A/Luminal B tumors and survival events across folds.

Saves:

* cv_fold_assignments.csv
* cv_fold_distribution.csv
* cv_fold_proportions.csv

The saved fold assignments will be reused for all later survival models so that
the mRNA-only baseline and integrated methylation + mRNA model are evaluated on
the same patient splits.


--------------------------------------------------------------------------------

08_Methylation_KNN_Imputation_For_Plotting.ipynb /
08_methylation_knn_imputation_For_Plotting.py:

Purpose: Perform exploratory KNN imputation of PAM50 promoter methylation beta
values and generate imputation diagnostics for Week 2 analysis.

Loads:

* meth_pam50.csv

KNN imputation:

* identifies and removes CpGs with no observed beta values in any patient because
  they cannot be imputed from the available data
* retains all remaining biologically selected promoter CpGs without variance-based
  feature filtering
* records methylation missingness per patient and per CpG before imputation
* evaluates distance-weighted KNN imputation with 3, 5, and 10 neighbors by
  randomly masking 2% of observed beta values and comparing reconstructed values
  with their known originals
* selects 10 neighbors because it produced the lowest reconstruction error
  (MAE and RMSE)
* imputes the remaining missing beta values using distance-weighted KNN with
  k = 10
* verifies that all missing values are filled and that imputed beta values remain
  within the expected 0–1 range

Generates and saves:

* methylation missingness tables before imputation
* KNN neighbor sensitivity comparison table
* KNN imputation summary table
* before/after methylation distribution diagnostic
* observed-versus-imputed beta-value distribution diagnostic

Outputs:

* meth_pam50_knn_imputed.csv
* all_missing_cpgs_removed_before_knn.csv
* pam50_promoter_cpg_variation_summary.csv
* methylation_missingness_per_patient_before_imputation.csv
* methylation_missingness_per_cpg_before_imputation.csv
* knn_neighbor_sensitivity.csv
* knn_imputation_summary.csv

Important:
The full-cohort imputed methylation matrix is used only for Week 2 exploratory
analysis, methylation-expression correlations, and visualizations. It must not
be used directly for cross-validated survival-model evaluation because that
would allow information from test-fold patients to influence imputation.

--------------------------------------------------------------------------------

KNN_Imputation_Helper_Function.py:

Purpose: Provide fold-safe methylation preprocessing for later survival-model
cross-validation.

For each cross-validation fold, the helper function:

* subsets methylation data into training and test patients
* removes CpGs that are completely missing within the training fold
* fits distance-weighted KNN imputation using training patients only
* applies the fitted imputer to the corresponding test patients
* optionally standardizes methylation features using training-fold means and
  standard deviations only

This helper prevents data leakage during Week 2 and Week 3 survival-model
evaluation. The full-cohort imputed methylation file should not be used as input
for cross-validated model training or evaluation.

--------------------------------------------------------------------------------

09_CpG_Expression_Correlation.py:

Purpose: Quantify the relationship between promoter methylation and gene
expression for each PAM50 gene (Task 1).

Loads:

* meth_pam50_knn_imputed.csv (full-cohort imputed methylation, exploratory)
* rna_pam50.csv
* cpg_gene_map.csv

For every promoter CpG, computes the Spearman correlation between its imputed beta
value and the matched gene's log2 expression across patients, with
Benjamini-Hochberg FDR correction. Aggregates per gene (mean and most-negative
correlation) and identifies the gene with the strongest silencing signal (most
negative correlation), which stage 10 reuses to define methylation strata.

Generates:

* cpg_expression_spearman.csv (per-CpG correlations and q-values)
* gene_methylation_expression_correlation_summary.csv (per-gene summary)
* methylation_expression_correlation_by_gene.png (per-gene mean-rho bar chart)
* top_silencing_gene_scatter.png (methylation vs expression for the top CpG)

Note: uses the full-cohort imputed matrix because this is descriptive analysis,
not cross-validated model evaluation, so no leakage concern applies.

--------------------------------------------------------------------------------

09b_Subtype_Correlation_Comparison.py:

Purpose: Compare the promoter methylation-expression relationship between Luminal A
and Luminal B (pitch method 3), to test whether methylation-associated gene
regulation differs between the two subtypes.

Loads:

* meth_pam50_knn_imputed.csv
* rna_pam50.csv
* cpg_gene_map.csv
* labels_luminal_brca.csv

For each PAM50 gene, computes the mean Spearman correlation between promoter
methylation and expression separately within LumA and within LumB patients, and
reports the per-gene difference between subtypes.

Generates:

* correlation_by_subtype.csv (per-gene rho in LumA, rho in LumB, difference)
* methylation_expression_correlation_LumA_vs_LumB.png (per-gene LumA vs LumB
  scatter; points off the y = x line indicate subtype-specific regulation)

Note: LumB has far fewer patients than LumA, so its per-gene correlations are
noisier; the comparison is descriptive (no formal test of correlation differences).

--------------------------------------------------------------------------------

10_Survival_Curves.py:

Purpose: Kaplan-Meier overall-survival visualizations (Task 1).

Loads:

* survival_luminal_clean.csv
* meth_pam50_knn_imputed.csv
* cpg_gene_map.csv
* gene_methylation_expression_correlation_summary.csv

Plots overall survival by PAM50 subtype (LumA vs LumB) and by high vs low promoter
methylation (median split) of the strongest silencing gene from stage 09, each
with a log-rank test.

Generates:

* km_survival_by_subtype.png
* km_survival_by_<gene>_methylation.png
* survival_logrank_tests.csv

--------------------------------------------------------------------------------

11_LASSO_Cox_mRNA_Baseline.py:

Purpose: Establish the mRNA-only LASSO-Cox survival benchmark (Task 2).

Loads:

* rna_pam50.csv
* survival_luminal_clean.csv
* cv_fold_assignments.csv

Fits a LASSO-penalized Cox proportional-hazards model on the log2-normalized PAM50
mRNA features using the shared 5-fold cross-validation splits. The model uses
scikit-survival's CoxnetSurvivalAnalysis (the same engine and nested
cross-validation strategy as the multi-omics model in stage 12, so the two
LASSO C-indices are directly comparable). Nested cross-validation is used:
the outer loop evaluates the model on the shared fold assignments, while an
inner 5-fold cross-validation selects the optimal L1 penalty (alpha).
Feature scaling and penalty selection are performed using the training
patients of each fold only, preventing data leakage. Reports the per-fold
concordance index (C-index), the selected alpha, and the number of genes
retained by the LASSO model.

Requires scikit-survival (pip install scikit-survival).

Output:

* lasso_cox_cv_results.csv (per-fold alpha, total genes, selected genes, test C-index)

--------------------------------------------------------------------------------

12_LASSO_Cox_Multiomics.py:

Purpose: Integrated LASSO-Cox survival model using both omics layers (mRNA
expression + promoter methylation), to test whether methylation improves on the
mRNA-only benchmark.

Loads:

* rna_pam50.csv
* meth_pam50.csv (raw beta values, with missing values)
* survival_luminal_clean.csv
* cv_fold_assignments.csv
* KNN_Imputation_Helper_Function.py

For each cross-validation fold, methylation is imputed using fold-safe KNN
imputation (fit on the training patients only), converted from beta values to
M-values (log2(beta/(1-beta))), standardized, and concatenated with
standardized mRNA expression features. The LASSO-penalized Cox model is fit
using scikit-survival's CoxnetSurvivalAnalysis. Nested cross-validation is
used: the outer loop evaluates model performance on the held-out fold, while
an inner 5-fold cross-validation selects the optimal L1 penalty (alpha).
The same shared fold assignments are used as the mRNA-only baseline so the
resulting C-indices are directly comparable.

Requires scikit-survival (pip install scikit-survival).

Output:

* lasso_cox_multiomics_cv_results.csv (per-fold alpha, total features, selected features, test C-index)

--------------------------------------------------------------------------------

13_LASSO_Cox_Model_Comparison.py:

Creates a quick evaluation of the results tables from the baseline and the integrated
LASSO Cox Model. Also saves 3 figues in results/figures for visualization of the results.
Also a equivalent Notebook was created in notebooks/ for exploration. 

Input:

* lasso_cox_cv_results
* lasso_cox_multiomics_cv_results

Output:

* cindex_boxplot.png
* cindex_mean_sd.png
* feature_selection_comparison.png

--------------------------------------------------------------------------------

NN_Cox_mRNA_Expression.py:

Purpose: Helper module implementing the neural-network Cox proportional hazards model for mRNA 
expression only.

Provides reusable functions for:

- constructing the mRNA-only neural network architecture
- computing the Cox partial likelihood loss
- training the model with early stopping
- evaluating the concordance index (C-index)
- predicting patient-level relative risk scores for ranking survival risk

This script is a helper module and is not intended to be run directly. It is imported by the tuning 
and final evaluation scripts.

--------------------------------------------------------------------------------

NN_Cox_Integrated.py:

Purpose: Helper module implementing the integrated neural-network Cox proportional hazards model using 
both mRNA expression and promoter methylation.

Provides reusable functions for:

- constructing the integrated neural network architecture
- combining expression and methylation features
- computing the Cox partial likelihood loss
- training with early stopping
- evaluating the concordance index (C-index)
- predicting patient-level relative risk scores for ranking survival risk

This script is a helper module and is not intended to be run directly. It is imported by the tuning and 
final evaluation scripts.

--------------------------------------------------------------------------------

14_tune_mRNA.py:

Purpose: Tune the hyperparameters of the mRNA-only neural-network Cox survival model.

Loads:

* NN_Cox_mRNA_Expression.py

Uses the shared cross-validation workflow from `NN_Cox_mRNA_Expression.py` to evaluate a grid of 
conservative neural-network hyperparameter combinations.

Tunes:

* learning rate
* weight decay
* hidden layer architecture
* dropout rate

Uses fixed values for:

* batch normalization: False
* batch size: 32
* early stopping patience: 10

For each configuration, the script records train, validation, and test C-index across the cross-validation 
folds. The final configuration is selected using a conservative 1-standard-deviation rule: among models 
within 1 SD of the best mean validation C-index, the script chooses the simplest model with the lowest 
overfitting gap.

Outputs:

* nn_mRNA_Only_conservative_joint_tuning_summary.csv
* nn_mRNA_Only_conservative_joint_tuning_folds.csv
* selected best hyperparameter configuration printed to the console

--------------------------------------------------------------------------------

15_tune_integrated_nn.py:

Purpose: Tune the hyperparameters of the integrated neural-network Cox survival model using both 
mRNA expression and promoter methylation.

Loads:

* NN_Cox_Integrated.py

Uses the shared cross-validation workflow from `NN_Cox_Integrated.py` to evaluate a grid of conservative 
multi-omics neural-network hyperparameter combinations.

Tunes:

* learning rate
* weight decay
* hidden layer architecture
* dropout rate

Uses fixed values for:

* batch normalization: False
* batch size: 32
* early stopping patience: 10
* methylation variance threshold: 0.0005

For each configuration, the script records train, validation, and test C-index across the cross-validation 
folds. The final configuration is selected using a conservative 1-standard-deviation rule: among models 
within 1 SD of the best mean validation C-index, the script chooses the simplest model with the lowest 
overfitting gap.

Outputs:

* nn_integrated_conservative_joint_tuning_summary.csv
* nn_integrated_conservative_joint_tuning_folds.csv
* selected best hyperparameter configuration printed to the console

--------------------------------------------------------------------------------

16_Run_best_Model_mRNA_Only.py:

Purpose: Train and evaluate the final mRNA-only neural-network Cox survival model using the best 
hyperparameter configuration identified during tuning.

Loads:

* NN_Cox_mRNA_Expression.py

Uses the selected hyperparameter configuration to train the mRNA-only neural-network survival model on the 
shared cross-validation folds.

The model uses:

* learning rate = 0.001
* weight decay = 0.001
* hidden layer architecture = [16]
* dropout rate = 0.6
* batch normalization = False
* batch size = 32
* early stopping patience = 10

Reports:

* train C-index
* validation C-index
* test C-index
* number of training epochs for each fold

Outputs:

* nn_mRNA_only_best_model_folds.csv
* nn_mRNA_only_best_model_summary.csv

--------------------------------------------------------------------------------

17_Run_Best_Model_Integrated.py:

Purpose: Train and evaluate the final integrated neural-network Cox survival model using the best 
hyperparameter configuration identified during tuning.

Loads:

* NN_Cox_Integrated.py

Uses the selected hyperparameter configuration to train the integrated neural-network survival model on the 
shared cross-validation folds.

The model uses:

* learning rate = 0.001
* weight decay = 0.03
* hidden layer architecture = [8]
* dropout rate = 0.5
* batch normalization = False
* methylation variance threshold = 0.0005
* batch size = 32
* early stopping patience = 10

Reports:

* train C-index
* validation C-index
* test C-index
* number of training epochs for each fold

Outputs:

* nn_integrated_best_model_folds.csv
* nn_integrated_best_model_summary.csv

Additionally saves the best-fold model weights and feature matrix for SHAP analysis
(handled inside NN_Cox_Integrated.py):

* nn_cox_integrated_best_fold_weights.pt
* nn_cox_integrated_shap_input.npy
* nn_cox_integrated_feature_names.csv

--------------------------------------------------------------------------------

18_Model_Comparison_Risk_Stratification.py:

Purpose: Compare all four survival models on the shared 5-fold cross-validation splits
and evaluate clinical risk stratification via Kaplan-Meier curves.

Loads:

* lasso_cox_cv_results.csv (script 11)
* lasso_cox_multiomics_cv_results.csv (script 12)
* nn_mRNA_only_best_model_folds.csv (script 16)
* nn_integrated_best_model_folds.csv (script 17)
* lasso_cox_mrna_risk_scores.csv (script 11)
* lasso_cox_multiomics_risk_scores.csv (script 12)
* nn_mrna_only_risk_scores.csv (script 16)
* nn_integrated_risk_scores.csv (script 17)
* survival_luminal_clean.csv

Section 1 — Model Performance Comparison (C-index):

* Performance summary table and mean C-index +/- SD plot across all four models
* C-index boxplot and per-fold bar chart showing fold-level variance
* LASSO feature selection stability across folds
* Pairwise Wilcoxon signed-rank tests between model pairs

Section 2 — Kaplan-Meier Risk Stratification:

* For each model, patients are split into predicted high- and low-risk groups
  at the median risk score, and Kaplan-Meier curves are drawn for each group
* Log-rank test assesses whether the risk groups show significantly different
  survival trajectories

Section 3 — Final Summary Table:

* Combined table of mean C-index, SD, and KM log-rank p-value per model

Generates:

* cindex_all_models_boxplot.png
* cindex_all_models_mean_sd.png
* cindex_per_fold_all_models.png
* feature_selection_stability.png
* km_risk_groups_mrna_only_lasso.png
* km_risk_groups_multi_omics_lasso.png
* km_risk_groups_mrna_only_nn.png
* km_risk_groups_multi_omics_nn.png
* all_models_performance_summary.csv
* wilcoxon_model_comparisons.csv
* km_risk_group_logrank.csv
* final_model_comparison_table.csv

Note: risk score files are saved by the updated versions of scripts 11, 12, 16
and 17 (risk score saving block added to each script's CV loop).

Run after scripts 11, 12, 16 and 17.

--------------------------------------------------------------------------------

19_SHAP_Analysis.py:

Purpose: Compute SHAP (SHapley Additive exPlanations) values for the best-fold
Multi-omics NN Cox model to quantify each feature's contribution to individual
patient risk predictions and identify the most important methylation and
expression features.

Loads:

* nn_cox_integrated_best_fold_weights.pt (saved by script 17)
* nn_cox_integrated_shap_input.npy (saved by script 17)
* nn_cox_integrated_feature_names.csv (saved by script 17)
* cpg_gene_map.csv

Rebuilds the exact network architecture from script 17 (num_nodes=[8],
dropout=0.5, batch_norm=False) and loads the saved weights. Uses
shap.GradientExplainer with a random background sample of 100 training
patients to compute SHAP values for all features and all training patients.

Generates:

* SHAP beeswarm plot showing the top 20 features and per-patient impact direction
* SHAP bar chart showing top 30 features colored by RNA vs methylation
* SHAP dependence plots for the top 3 features

Saves:

* shap_values_nn_integrated.csv (full SHAP matrix, patients x features)
* shap_feature_importance.csv (features ranked by mean absolute SHAP)
* shap_omics_importance_summary.csv (total importance split by RNA vs methylation)
* shap_beeswarm_top20.png
* shap_bar_top30.png
* shap_dependence_top3.png

Note: the saved shap_feature_importance.csv can be used directly for downstream
pathway enrichment analysis by extracting gene names from the top-ranked features.

Requires shap (pip install shap). Run after script 17.

--------------------------------------------------------------------------------

How to run:

Run the R scripts in numerical order (01 -> 02 -> 03 -> 04 -> 05 -> 05b) from the
repository root. Script 01 downloads the raw TCGA data (several GB) and only needs
to be run once. Script 05b exports the CpG -> gene map used by the correlation
analysis. The data/ folder is git-ignored, so the .RDS files remain local and are
not pushed to the repository.

Next, run the exploratory Python scripts:

06_week1_eda.py ->
06b_EDA_Extension.py ->
07_create_cv_folds.py ->
08_Methylation_KNN_Imputation_For_Plotting.py ->
09_CpG_Expression_Correlation.py ->
09b_Subtype_Correlation_Comparison.py ->
10_Survival_Curves.py

LASSO-Cox survival models:

11_LASSO_Cox_mRNA_Baseline.py ->
12_LASSO_Cox_Multiomics.py

Neural-network Cox survival models:

14_tune_mRNA.py ->
15_tune_integrated_nn.py ->
16_Run_best_Model_mRNA_Only.py ->
17_Run_Best_Model_Integrated.py

Model comparison and risk stratification:

18_Model_Comparison_Risk_Stratification.py

SHAP feature importance analysis:

19_SHAP_Analysis.py

Note: scripts 11, 12, 16 and 17 must include the risk score saving block in their
CV loop before running script 18. Script 19 requires the model weights saved by
script 17 and pip install shap before running.

Helper modules:

* KNN_Imputation_Helper_Function.py
* NN_Cox_mRNA_Expression.py
* NN_Cox_Integrated.py

These helper modules are imported by the survival-model scripts and should not be
run directly.
