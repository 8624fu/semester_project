# Clean Environment ------------------------------------------------------------

rm(list = ls())

# ------------------------------------------------------------------------------

# Load Packages

library(TCGAbiolinks)
library(sesame)
library(sesameData)
library(SummarizedExperiment)
library(here)

here::i_am("scripts/01_Initial_Data_Download.R")

# Configure GDC Data Transfer Client -------------------------------------------

# Users may optionally define the location of gdc-client before running:
#
# Sys.setenv(
#   GDC_CLIENT_PATH = "/full/path/to/gdc-client"
# )
#
# The script also checks several common installation locations automatically.

find_gdc_client <- function() {
  
  # 1. Check whether gdc-client is already available through PATH
  client_from_path <- Sys.which("gdc-client")
  
  if (nzchar(client_from_path)) {
    return(unname(client_from_path))
  }
  
  # 2. Check an optional user-defined environment variable
  client_from_env <- Sys.getenv("GDC_CLIENT_PATH")
  
  # 3. Check common locations
  candidate_paths <- c(
    client_from_env,
    path.expand("~/Downloads/gdc-client"),
    path.expand("~/bin/gdc-client"),
    "/opt/homebrew/bin/gdc-client",
    "/usr/local/bin/gdc-client"
  )
  
  candidate_paths <- candidate_paths[nzchar(candidate_paths)]
  
  existing_paths <- candidate_paths[file.exists(candidate_paths)]
  
  if (length(existing_paths) == 0) {
    stop(
      paste(
        "Could not find the GDC Data Transfer Tool.",
        "",
        "Install gdc-client and either:",
        "1. Add it to your system PATH, or",
        "2. Set GDC_CLIENT_PATH to its full location.",
        "",
        'Example:',
        'Sys.setenv(GDC_CLIENT_PATH = "~/Downloads/gdc-client")',
        sep = "\n"
      )
    )
  }
  
  normalizePath(existing_paths[1])
}


gdc_client <- find_gdc_client()

# Add the directory containing gdc-client to PATH so that
# TCGAbiolinks can call it by name.
gdc_client_dir <- dirname(gdc_client)

Sys.setenv(
  PATH = paste(
    gdc_client_dir,
    Sys.getenv("PATH"),
    sep = .Platform$path.sep
  )
)

# Verify that the client is now accessible
if (!nzchar(Sys.which("gdc-client"))) {
  stop("gdc-client was found, but could not be added to PATH.")
}

message("Using GDC client: ", Sys.which("gdc-client"))

client_status <- system2(
  "gdc-client",
  "--version"
)

if (!identical(client_status, 0L)) {
  stop("gdc-client was found, but the version check failed.")
}

# Download Helper Function -----------------------------------------------------

download_gdc_query <- function(query, directory, dataset_name) {
  
  message("")
  message("Starting ", dataset_name, " download...")
  
  tryCatch(
    {
      GDCdownload(
        query = query,
        method = "client",
        directory = directory
      )
      
      message(dataset_name, " download completed successfully.")
      
      invisible(TRUE)
    },
    
    error = function(e) {
      message("")
      message("============================================================")
      message(dataset_name, " download failed.")
      message("Reason: ", conditionMessage(e))
      message("")
      message("Successfully downloaded files have been kept.")
      message("Rerun this script once the connection is available.")
      message("gdc-client will skip completed files and resume the download.")
      message("============================================================")
      message("")
      
      quit(
        save = "no",
        status = 1,
        runLast = FALSE
      )
    }
  )
}

# Create Directories -----------------------------------------------------------

dirs <- c(
  here("data", "raw"),
  here("data", "multiomics"),
  here("data", "GDCdata"))

for (d in dirs) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)
  }
}

gdc_data_dir <- here("data", "GDCdata")

# Increase macOS vector-memory limit when necessary ----------------------------

if (Sys.info()[["sysname"]] == "Darwin") {
  mem.maxVSize(32000)
  message("R vector-memory limit: ", mem.maxVSize(), " MB")
}

# Methylation - Initial Data Download ------------------------------------------

meth_query <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "DNA Methylation",
  data.type = "Methylation Beta Value",
  platform = "Illumina Human Methylation 450"
)

meth_rds <- here("data", "raw", "meth_tcga_brca.rds")

if (file.exists(meth_rds)) {
  message(
    "Prepared methylation RDS already exists. ",
    "Skipping methylation download and preparation."
  )
} else {
  download_gdc_query(
    query = meth_query,
    directory = gdc_data_dir,
    dataset_name = "Methylation"
  )
  
  message("Preparing methylation data...")
  
  meth <- GDCprepare(
    query = meth_query,
    directory = gdc_data_dir
  )
  
  saveRDS(meth, meth_rds)
  
  rm(meth)
  gc()
  
  message("Methylation data prepared and saved successfully.")
}

# RNA - Initial Data Download -------------------------------------------------- 

rna_query <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)

rna_rds <- here("data", "raw", "rna_tcga_brca.rds")

if (file.exists(rna_rds)) {
  message(
    "Prepared RNA RDS already exists. ",
    "Skipping RNA download and preparation."
  )
} else {
  download_gdc_query(
    query = rna_query,
    directory = gdc_data_dir,
    dataset_name = "RNA"
  )

  message("Preparing RNA data...")

  rna <- GDCprepare(
    query = rna_query,
    directory = gdc_data_dir
  )

  saveRDS(rna, rna_rds)

  rm(rna)
  gc()

  message("RNA data prepared and saved successfully.")
}

# Clinical - Initial Data Download ---------------------------------------------

message("")
message("Downloading clinical data...")

clinical <- tryCatch(
  {
    GDCquery_clinic(
      project = "TCGA-BRCA",
      type = "clinical"
    )
  },
  
  error = function(e) {
    message("")
    message("============================================================")
    message("Clinical data download failed.")
    message("Reason: ", conditionMessage(e))
    message("")
    message("Rerun this script once the connection is available.")
    message("============================================================")
    message("")
    
    quit(
      save = "no",
      status = 1,
      runLast = FALSE
    )
  }
)

message("Clinical data downloaded successfully.")

saveRDS(clinical, here("data", "raw", "clinical_brca.rds"))

rm(clinical)
gc()

message("Clinical data downloaded and saved successfully.")

# Clean Up Temporary Files -----------------------------------------------------

temp_files <- c(
  "gdc_manifest.txt",
  "gdc_client_configuration.dtt",
  "df.rds",
  "results.rds"
)

temp_files <- temp_files[file.exists(temp_files)]

if (length(temp_files) > 0) {
  file.remove(temp_files)
  message("Temporary download files removed.")
}