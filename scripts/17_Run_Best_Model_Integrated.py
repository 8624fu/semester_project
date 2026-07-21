import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "scripts"))


import json
import pandas as pd

from NN_Cox_Integrated import run_cv


OUT_DIR = PROJECT_ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = OUT_DIR / "nn_integrated_best_hyperparameters.json"


def load_best_config():
    """Load the hyperparameters selected by the integrated-model tuning script."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Best hyperparameter file not found: {CONFIG_FILE}\n"
            "Run scripts/15_tune_integrated_nn.py before running this script."
        )
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["evaluate_on_test"] = True
    return config   

def summarize_cv(cv, config):
    summary_config = config.copy() 
    summary_config["num_nodes"] = json.dumps(summary_config["num_nodes"])

    return pd.DataFrame([{
        **summary_config,
        "mean_train_c_index": cv["train_c_index"].mean(),
        "sd_train_c_index": cv["train_c_index"].std(),
        "mean_val_c_index": cv["val_c_index"].mean(),
        "sd_val_c_index": cv["val_c_index"].std(),
        "mean_test_c_index": cv["test_c_index"].mean(),
        "sd_test_c_index": cv["test_c_index"].std(),
        "mean_epochs_trained": cv["epochs_trained"].mean(),
    }])


def main():
    best_config = load_best_config()
    print("Running final integrated NN-Cox model with best configuration:")
    print(json.dumps(best_config, indent=2))

    cv = run_cv(**best_config)

    summary = summarize_cv(cv, best_config)

    cv.to_csv(
        OUT_DIR / "nn_integrated_best_model_folds.csv",
        index=False,
    )

    summary.to_csv(
        OUT_DIR / "nn_integrated_best_model_summary.csv",
        index=False,
    )

    print("\nFold-level results:")
    print(cv.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()