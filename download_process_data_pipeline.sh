#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Running R setup..."
Rscript scripts/00_install_packages_for_R.R

echo "Running R data preparation..."
# Rscript scripts/01_Initial_Data_Download.R
Rscript scripts/02_Merge_Data.R
Rscript scripts/03_Check_Missings.R
Rscript scripts/04_Feature_Selection.R
Rscript scripts/04b_check_stage.R
Rscript scripts/05_export_for_python.R
Rscript scripts/05b_export_cpg_gene_map.R

echo "Pipeline completed successfully."