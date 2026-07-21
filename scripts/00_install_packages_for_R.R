# 00_install_packages.R
# Install all R packages required by the project.
# Run once before executing the analysis pipeline:
# Rscript scripts/00_install_packages.R


# CRAN packages ---------------------------------------------------------------

cran_packages <- c(
  "here",
  "glue",
  "dplyr"
)

missing_cran <- cran_packages[
  !vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_cran) > 0) {
  message(
    "Installing missing CRAN packages: ",
    paste(missing_cran, collapse = ", ")
  )
  
  install.packages(
    missing_cran,
    repos = "https://cloud.r-project.org"
  )
} else {
  message("All required CRAN packages are already installed.")
}


# Bioconductor package manager ------------------------------------------------

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  message("Installing BiocManager.")
  
  install.packages(
    "BiocManager",
    repos = "https://cloud.r-project.org"
  )
}


# Bioconductor packages -------------------------------------------------------

bioconductor_packages <- c(
  "TCGAbiolinks",
  "SummarizedExperiment",
  "sesameData",
  "sesame",
  "DESeq2",
  "org.Hs.eg.db",
  "minfi",
  "IlluminaHumanMethylation450kanno.ilmn12.hg19"
)

missing_bioconductor <- bioconductor_packages[
  !vapply(
    bioconductor_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_bioconductor) > 0) {
  message(
    "Installing missing Bioconductor packages: ",
    paste(missing_bioconductor, collapse = ", ")
  )
  
  BiocManager::install(
    missing_bioconductor,
    ask = FALSE,
    update = FALSE
  )
} else {
  message("All required Bioconductor packages are already installed.")
}


# Installation check ----------------------------------------------------------

all_packages <- c(cran_packages, bioconductor_packages)

failed_packages <- all_packages[
  !vapply(all_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(failed_packages) > 0) {
  stop(
    "The following packages could not be installed: ",
    paste(failed_packages, collapse = ", ")
  )
}

message("R package setup completed successfully.")

# External software check ------------------------------------------------------

gdc_client <- Sys.which("gdc-client")

if (!nzchar(gdc_client)) {
  warning(
    paste(
      "The GDC Data Transfer Tool was not found on the system PATH.",
      "It is required by scripts/01_Initial_Data_Download.R",
      "for downloading the large TCGA molecular datasets.",
      "",
      "Download the appropriate version from the official GDC website",
      "and either add it to PATH or set the GDC_CLIENT_PATH",
      "environment variable to its full location.",
      sep = "\n"
    ),
    call. = FALSE
  )
} else {
  message("GDC Data Transfer Tool found: ", gdc_client)
}