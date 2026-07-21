# 05_export_for_python.R
# Export the PAM50 multi-omics dataset from R format to Python-friendly CSV files.

rm(list = ls())

library(dplyr)
library(here)

here::i_am("scripts/05_export_for_python.R")

# Create output folder 
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

# Load pam50 features
pam50_features <- readRDS(here("data", "multiomics", "pam50_features_brca.rds"))

# Pull out each component
rna_pam50 <- pam50_features$rna
meth_pam50 <- pam50_features$meth
clinical <- pam50_features$clinical
y <- pam50_features$y

# Convert full TCGA barcodes to patient IDs
colnames(rna_pam50) <- substr(colnames(rna_pam50), 1, 12)
colnames(meth_pam50) <- substr(colnames(meth_pam50), 1, 12)

# Check that RNA and methylation patients are in identical order
stopifnot(identical(colnames(rna_pam50), colnames(meth_pam50)))

# Make sure clinical data is in the same order
clinical$patient <- substr(clinical$patient, 1, 12)

stopifnot(identical(clinical$patient, colnames(rna_pam50)))

# Check labels are also aligned
stopifnot(length(y) == ncol(rna_pam50))

# Transpose so each row is one patient and each column is one feature
rna_python <- as.data.frame(t(rna_pam50))
meth_python <- as.data.frame(t(meth_pam50))

# Preserve patient IDs as a named column instead of only row names
rna_python$patient <- rownames(rna_python)
meth_python$patient <- rownames(meth_python)

# Put patient first
rna_python <- rna_python[, c("patient", setdiff(colnames(rna_python), "patient"))]
meth_python <- meth_python[, c("patient", setdiff(colnames(meth_python), "patient"))]

dim(rna_python)
dim(meth_python)

head(rna_python[, 1:6])
head(meth_python[, 1:6])

labels_python <- data.frame(
  patient = colnames(rna_pam50),
  subtype = as.character(y)
)

# Confirm all files have the same patient order
stopifnot(identical(rna_python$patient, meth_python$patient))
stopifnot(identical(rna_python$patient, labels_python$patient))

clinical_python <- clinical %>%
  dplyr::select(
    patient,
    BRCA_Subtype_PAM50,
    vital_status,
    year_of_death,
    days_to_death,
    days_to_last_follow_up
  )

stopifnot(identical(rna_python$patient, clinical_python$patient))

# wrote csv files
write.csv(
  rna_python,
  here("data", "processed", "rna_pam50.csv"),
  row.names = FALSE
)

write.csv(
  meth_python,
  here("data", "processed", "meth_pam50.csv"),
  row.names = FALSE
)

write.csv(
  clinical_python,
  here("data", "processed", "clinical_luminal_brca.csv"),
  row.names = FALSE
)

write.csv(
  labels_python,
  here("data", "processed", "labels_luminal_brca.csv"),
  row.names = FALSE
)

