# Clean Environment ------------------------------------------------------------

rm(list = ls())


# Load Multiomics Data ---------------------------------------------------------

library(here)

here::i_am("scripts/03_Check_Missings.R")

multiomics <- readRDS(here("data", "multiomics", "multiomics_luminal_brca.rds"))

beta <- multiomics$beta

rna_mat <- multiomics$rna

clinical_common <- multiomics$clinical

y <- multiomics$y

# ------------------------------------------------------------------------------