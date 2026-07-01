import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from pycox.models import CoxPH
warnings.simplefilter("ignore")
print("torch", torch.__version__)


# network architecture

NUM_NODES   = [32, 16]   
DROPOUT     = 0.4        
BATCH_NORM  = True       
OUTPUT_BIAS = False      
LEARNING_RATE = 0.01     
WEIGHT_DECAY  = 0.10     
BATCH_SIZE    = 64
MAX_EPOCHS    = 256
PATIENCE      = 15       
VAL_FRACTION  = 0.20    
RANDOM_STATE = 42        


DATA_DIR = Path("../data/processed") 

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)




# import data and make sure each patient is existent in each data frame.

rna   = pd.read_csv(DATA_DIR / "rna_pam50.csv").set_index("patient")
surv  = pd.read_csv(DATA_DIR / "survival_luminal_clean.csv").set_index("patient")
folds = pd.read_csv(DATA_DIR / "cv_fold_assignments.csv").set_index("patient")



# make sure each patient has a valid follow-up survival time

surv = surv[surv["time"].notna() & (surv["time"] > 0)]
patients = rna.index.intersection(surv.index).intersection(folds.index)

rna     = rna.loc[patients]
surv    = surv.loc[patients]
fold_id = folds.loc[patients, "fold"]

GENES = list(rna.columns)
print(f"Overview of Input Data: \n\nPatients: {len(patients)} | genes: {len(GENES)} | "
      f"events: {int(surv['event'].sum())} | folds: {sorted(fold_id.unique())}\n")




# Define helper functions: 

def make_xy(ids, scaler):
    # Scale features and build the pycox label.
    x = scaler.transform(rna.loc[ids]).astype("float32")
    y = (surv.loc[ids, "time"].values.astype("float32"),
         surv.loc[ids, "event"].values.astype("float32"))
    return x, y


def build_net(in_features):
    # Build the MLP risk network
    layers = []
    prev_units = in_features
    for n_units in NUM_NODES:
        layers.append(nn.Linear(prev_units, n_units))   
        layers.append(nn.ReLU())                         
        if BATCH_NORM:
            layers.append(nn.BatchNorm1d(n_units))      
        if DROPOUT > 0:
            layers.append(nn.Dropout(DROPOUT))           
        prev_units = n_units                             
    layers.append(nn.Linear(prev_units, 1, bias=OUTPUT_BIAS))
    return nn.Sequential(*layers)


def build_model(in_features):
    # DeepSurv = risk network + Cox partial-likelihood loss (CoxPH).
    net = build_net(in_features)
    optimizer = tt.optim.Adam(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model = CoxPH(net, optimizer)
    return model


def c_index(model, x, durations, events):
    # C-index on the predicted log-risk
    risk = model.predict(x).ravel()
    result = concordance_index_censored(events.astype(bool), durations, risk)
    return result[0] 



# cross validated training & evaluation

rows = []
for f in sorted(fold_id.unique()):
    # Outer split: test set.
    train_val_ids = fold_id.index[fold_id != f]
    test_ids      = fold_id.index[fold_id == f]

    # Create internal validation set out of the training patients for early
    # stopping only (stratified on event so both splits keep some deaths).
    tr_ids, val_ids = train_test_split(
        train_val_ids, test_size=VAL_FRACTION, random_state=RANDOM_STATE,
        stratify=surv.loc[train_val_ids, "event"])

    # Fold-safe scaling: fit to training patients only.
    scaler = StandardScaler().fit(rna.loc[tr_ids])
    x_tr,  y_tr  = make_xy(tr_ids,  scaler)
    x_val, y_val = make_xy(val_ids, scaler)
    x_te,  y_te  = make_xy(test_ids, scaler)
    durations_te, events_te = y_te 

    # Build and train DeepSurv with early stopping on the validation loss.
    torch.manual_seed(RANDOM_STATE)         
    model = build_model(x_tr.shape[1])
    log = model.fit(
        x_tr, y_tr,
        batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
        callbacks=[tt.callbacks.EarlyStopping(patience=PATIENCE)],
        val_data=(x_val, y_val), val_batch_size=BATCH_SIZE, verbose=False)

    # Evaluate on the held-out test fold with C-index.
    test_c_index = c_index(model, x_te, durations_te, events_te)

    epochs_trained = log.epoch + 1          
    rows.append({"fold": f, "n_test": len(test_ids),
                 "epochs_trained": epochs_trained, "test_c_index": test_c_index})
    print(f"Fold {f}: C-index={test_c_index:.3f} | epochs={epochs_trained}")


# summarize results 

cv = pd.DataFrame(rows)

out_dir = Path("../results/tables")
out_dir.mkdir(parents=True, exist_ok=True)
cv.to_csv(out_dir / "nn_cox_mrna_cv_results.csv", index=False)

mean_ci, sd_ci = cv["test_c_index"].mean(), cv["test_c_index"].std()
print(cv.to_string(index=False))
print(f"\nC-index of NN-Cox mRNA Expression only:\n{mean_ci:.3f} +/- {sd_ci:.3f} (5-fold CV)")