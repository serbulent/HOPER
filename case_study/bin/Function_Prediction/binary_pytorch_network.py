"""
- Module implements simple neural network with  4 hidden layer. 
-Module make predictions for model training and test. Module  draws training and validation loss for better understanding of model behavior
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

seed_plt=42
random.seed(seed_plt)
torch.manual_seed(seed_plt)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_plt)

import numpy as np
#from HoloProtRepAFPML import BinaryTrainModelsWithHyperParameterOptimization
from Function_Prediction import F_max_scoring
# Eğer kf bir liste ise: (safe_kfold, stratified_balanced_kfold)
from imblearn.over_sampling import SMOTE


class Net(nn.Module):
    def __init__(self, input_size, class_number):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        #self.fc5 = nn.Linear(32, 16)
        #self.fc6 = nn.Linear(16, 8)
        #self.fc7 = nn.Linear(8, 8)
        #self.fc8 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(32, class_number)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        #x = F.relu(self.fc6(x))
        #x = F.relu(self.fc7(x))
        #x = F.relu(self.fc8(x))
        x = self.fc5(x)
        return x


def NN(
    kf,
    path_train,
    protein_representation,
    model_label,
    input_size,
    representation_name,
    protein_and_representation_dictionary,
):
    #breakpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_max_cv_train = []
    f_max_cv_test = []

    model_label_pred_lst = []
    label_lst = []
    model_label_pred_test_lst = []
    label_lst_test = []

    protein_name_tr = []
    protein_name = []

    running_loss_lst_s = []
    val_loss_lst_s = []

    # -------------------------
    # VECTOR → PROTEIN MAP (O(1))
    # -------------------------
    vector_to_protein = {
        tuple(v): k for k, v in protein_and_representation_dictionary.items()
    }
    rows = []
  
    for fold_id, (fold_train_index, fold_test_index) in enumerate(kf.split(protein_representation, model_label)):

        # ======================
        # DATA SPLIT
        # ======================
        if isinstance(protein_representation, pd.DataFrame):
            protein_representation = protein_representation.values
        
        X_train = protein_representation[fold_train_index]
        X_test  = protein_representation[fold_test_index]
        
        # X_train: (138, 1) -> her hücrede (512,) vektör var
        X_train = np.vstack(X_train[:, 0])
        X_test  = np.vstack(X_test[:, 0])

        y_train = model_label[fold_train_index].astype(int)
        train_pos = np.sum(y_train == 1)
        train_neg = np.sum(y_train == 0)

        y_test  = model_label[fold_test_index].astype(int)

        # ======================
        # SAFE SMOTE
        # ======================
        n_pos = np.sum(y_train == 1)
        n_neg=len(y_train)-n_pos
        if n_pos > 3:
            k = min(3, n_pos - 1)
            smote = SMOTE(sampling_strategy={1:n_pos*2},k_neighbors=k, random_state=42)           
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
        # ======================
        # TORCH
        # ======================
        X_train_t = torch.tensor(X_train, dtype=torch.double).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.double).to(device)

        X_test_t = torch.tensor(X_test, dtype=torch.double).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.double).to(device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        torch.save(train_ds, os.path.join(path_train,f"{representation_name[0]}_train_dataset_fnn.pt"))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        # ======================
        # MODEL
        # ======================
        model = Net(input_size, 1).double().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        running_loss_lst = []
        val_loss_lst = []

        # ======================
        # TRAINING
        # ======================
        for epoch in range(500):
            model.train()
            epoch_loss = 0.0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            running_loss_lst.append(epoch_loss / len(train_loader))

            model.eval()
            with torch.no_grad():
                val_logits = model(X_test_t)
                val_loss = criterion(val_logits, y_test_t.unsqueeze(1))
                val_loss_lst.append(val_loss.item())

        running_loss_lst_s.append(running_loss_lst)
        val_loss_lst_s.append(val_loss_lst)
        # ======================
        # THRESHOLD SELECTION (TRAIN)
        # ======================
        with torch.no_grad():
            probs_train = torch.sigmoid(model(X_train_t)).squeeze().cpu().numpy()
       
        best_f, best_t = 0.0, 0.5
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs_train >= t).astype(int)
            f = F_max_scoring.evaluate_annotation_f_max(y_train, preds)
            if f > best_f:
                best_f, best_t = f, t
                breakpoint()
        preds_train = (probs_train >= best_t).astype(int)
        

       
        with torch.no_grad():
            probs_val = torch.sigmoid(model(X_test_t)).squeeze().cpu().numpy()
            y_val_np = y_test_t.cpu().numpy().astype(int)
        f_max_cv_test.append(
            F_max_scoring.evaluate_annotation_f_max(
                y_val_np, (probs_val >= best_t).astype(int)
            )
        )
        # ======================
        # TRAIN METRIC (INFO)
        # ======================
       
        #breakpoint()
        f_max_cv_train.append(best_f)

        model_label_pred_lst.append(preds_train)
        label_lst.append(y_train)

        model_label_pred_test_lst.append((probs_val >= best_t).astype(int))
        label_lst_test.append(y_val_np)
        test_proteins = [vector_to_protein.get(tuple(v)) for v in X_test]
    
        y_pred_test = (probs_val >= best_t).astype(int)
        
        pos_ids = [test_proteins[i] for i in np.where(y_val_np == 1)[0]]
        neg_ids = [test_proteins[i] for i in np.where(y_val_np == 0)[0]]
        fp_ids  = [test_proteins[i] for i in np.where((y_pred_test == 1) & (y_val_np == 0))[0]]
        fn_ids  = [test_proteins[i] for i in np.where((y_pred_test == 0) & (y_val_np == 1))[0]]

        # ======================
        # PROTEIN NAMES
        # ======================
        for vec in X_train:
            protein_name_tr.append(vector_to_protein.get(tuple(vec)))

        for vec in X_test:
            protein_name.append(vector_to_protein.get(tuple(vec)))
        rows.append({
            "Fold": fold_id,
            "F_max": f_max_cv_test[-1],
            "Threshold": best_t,
            "Train_Pos(1)": train_pos,
            "Train_Neg(0)": train_neg,
            "Test_Pos(1)": len(pos_ids),
            "Test_Neg(0)": len(neg_ids),
            "False_Pos_IDs": ",".join(fp_ids),
            "False_Neg_IDs": ",".join(fn_ids),
            "Pos_IDs": ",".join(pos_ids),
            "Neg_IDs": ",".join(neg_ids),
        })
    parameter = {
        "classifier": "NN",
        "representation": representation_name,
        "optimizer": "Adam",
        "lr": 1e-3,
        "epochs": 500,
        "loss": "BCEWithLogitsLoss",
        "threshold": "val_fmax",
    }
    
    df_folds = pd.DataFrame(rows)
    df_folds.to_csv(
        os.path.join(path_train, f"{representation_name[0]}_FNN_fold_results.csv"),
        index=False
    )


    return (
        f_max_cv_train,
        f_max_cv_test,
        model,
        model_label_pred_lst,
        label_lst,
        protein_name_tr,
        parameter,
        protein_name,
        model_label_pred_test_lst,
        label_lst_test,
    )

 