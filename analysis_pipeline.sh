#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Running Python setup..."
python scripts/00b_install_python_packages.py

echo "Running exploratory analyses..."
python scripts/06_Methylation_EDA.py
python scripts/06b_week1_eda.py
python scripts/07_create_cv_folds.py
python scripts/08_methylation_knn_imputation_For_Plotting.py
python scripts/08b_EDA_Extension.py
python scripts/09_CpG_Expression_Correlation.py
python scripts/09b_Subtype_Correlation_Comparison.py
Rscript scripts/09c_Differential_Methylation.R
Rscript scripts/09d_Subtype_Coupling_FisherZ.R
python scripts/10_Survival_Curves.py

echo "Running survival models..."
python scripts/11_LASSO_Cox_mRNA_Baseline.py
python scripts/12_LASSO_Cox_Multiomics.py
python scripts/13_LASSO_Cox_Model_Comparison.py
python scripts/14_tune_mRNA.py
python scripts/15_tune_integrated_nn.py
python scripts/16_Run_best_Model_mRNA_Only.py
python scripts/17_Run_Best_Model_Integrated.py
python scripts/18_Model_Comparison_Risk_Stratification.py

echo "Running feature-importance analyses..."
python scripts/19a_LASSO_Feature_Importance.py
python scripts/19b_NN_SHAP_Feature_Importance.py
python scripts/20a_Feature_Importance_Overview.py
python scripts/20b_Gene_Modality_Matrix.py
python scripts/21a_Functional_Annotation.py
python scripts/21b_Pathway_Enrichment_PAM50.py
python scripts/22_Post_Feedback_Models.py

echo "Pipeline completed successfully."