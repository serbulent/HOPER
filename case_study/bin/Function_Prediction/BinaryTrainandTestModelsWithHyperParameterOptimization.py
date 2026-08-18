"""
This module trains and test protein function models and reveals best model and hyperparameters. The module structure is the following:

- The module implements ``check_for_at_least_two_class_sample_exits`` method. The method takes a dataframes as input.
The input dataframe has varying number of columns. Each column represent a class (i.e. GO ids). 
The methods analyze the data frame to control at least two positive sample exits for each class.

- The module implements ``select_best_model_with_hyperparameter_tuning`` method. The method takes representation name, a dataframe list and 
scoring function,list of preferred model names as input. The dataframe has 3 columns 'Label','Entry' and 'Vector'. The method 
trains models and search for best model and hyperparameters. Then module test modules

- The module implements ``binary_evaluate`` method. For calculation of model metrics.


"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
    matthews_corrcoef
)

from sklearn.metrics import make_scorer
import ast
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)

from sklearn.metrics import multilabel_confusion_matrix
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer
from sklearn import metrics

path = os.getcwd()
sys.path.append(path + "/case_study/bin/")
from Function_Prediction import binary_pytorch_network
from Function_Prediction.binary_pytorch_network import NN
from Function_Prediction import binary_prediction
from Function_Prediction import binary_evaluate
import torch
import joblib
random_state=42
from sklearn.metrics import make_scorer
from Function_Prediction.Model_Parameters import Kneighbors_Classifier_parameters
from Function_Prediction.Model_Parameters import SVC_Classifier_parameters
from Function_Prediction.Model_Parameters import RandomForest_Classifier_parameters
from Function_Prediction import F_max_scoring
import random
import os, random

import torch
import openpyxl
from openpyxl import Workbook


from xgboost import XGBClassifier

from sklearn.metrics import make_scorer

def sensitivity_score(y_true, y_pred):
    # Sensitivity = Recall = TP / (TP + FN)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def specificity_score(y_true, y_pred):
    # Specificity = TN / (TN + FP)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

sensitivity_scorer = make_scorer(sensitivity_score)
specificity_scorer = make_scorer(specificity_score)


def save_fold_metrics_xlsx(
    save_path,
    representation_name,
    classifier_name,
    fmax_list,
    train_pos_list,
    train_neg_list,
    test_pos_list,
    test_neg_list,
    eval_type
):
    """
    TRAIN ve TEST fold dağılımını TEK Excel dosyasında kaydeder.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = f"{eval_type}_metrics"

    ws.append([
        "Representation", "Classifier", "Fold",
        "F_max", "F_max_STD",
        "Train_Pos(1)", "Train_Neg(0)",
        "Test_Pos(1)", "Test_Neg(0)"
    ])

    fmax_std = np.std(fmax_list)

    for i in range(len(fmax_list)):
        ws.append([
            representation_name[0],
            classifier_name,
            i,
            fmax_list[i],
            fmax_std,
            train_pos_list[i],
            train_neg_list[i],
            test_pos_list[i],
            test_neg_list[i]
        ])

    filename = os.path.join(
        save_path,
        f"{representation_name[0]}_{classifier_name}_{eval_type}_fold_metrics.xlsx"#smote_
    )
    wb.save(filename)
    print(f"Saved metrics Excel: {filename}")

def set_seed(seed: int = 42, deterministic: bool = True):
    # Python & OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # çoklu GPU

    # CuDNN / determinism
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.12+ için:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
set_seed(42)
# if every fold contains at least 2 positive samples return true,otherwise return false



def neural_network_eval(
    f_max_cv,
    kf,
    model,
    model_label_pred_lst,
    label_lst,
    index_,
    representation_name,
    classifier_name,
    file_name,
    eval_type,
    protein_name,
    path,
    parameter,
):
    #breakpoint()
    representation_name_concated = representation_name
    #breakpoint()
    if eval_type=="training":
      #breakpoint()
      paths =os.path.join(path,"training",representation_name_concated[0]+"_"+classifier_name+"_binary_classifier.pt")
      #breakpoint()
      torch.save(model.state_dict(), paths)
      
      
      best_parameter_dataframe = pd.DataFrame(parameter)
      training_path=os.path.join(path,"training","Neural_network_"+ representation_name_concated[0]+"_binary_classifier_best_parameter.csv")
      best_parameter_dataframe.to_csv(training_path,index=False)
     
    
      
    binary_evaluate.evaluate(
        kf,
        model_label_pred_lst,
        label_lst,
        f_max_cv,
        classifier_name,
        representation_name_concated[0],
        file_name,
        index_,
        eval_type,
    )

    label_predictions = pd.DataFrame(
        np.concatenate(model_label_pred_lst), columns=["Label"]
    )
    label_prediction_path=os.path.join(path,eval_type,representation_name_concated[0]+"_binary_classifier_"+classifier_name+eval_type+"_predictions.csv")
    label_predictions.insert(0, "protein_id", protein_name)
    label_predictions.to_csv(
        label_prediction_path,
        index=False,
    )


best_param_list = []
def check_for_at_least_two_class_sample_exits(y):

    for column in list(y):
        column_sum = np.sum(y[column])
        if column_sum < 2:
            print(
                "At least 2 positive samples needed for each class {0} class has {1} positive samples".format(
                    column, column_sum
                )
            )
            return False
    return True


best_param_list = []

from sklearn.metrics import make_scorer, roc_auc_score, matthews_corrcoef, f1_score



import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE



best_param_list = []


def fmax_scorer_func(y_true, y_prob):
    thresholds = np.linspace(0,1,100)
    best = 0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best:
            best = f1
    return best


def save_fold_class_counts(grid, X, y, save_path, rep_name, classifier_name):
    """
    GridSearchCV içindeki her fold için class dağılımını hesaplar ve Excel'e kaydeder.
    """
    fold_records = []

    # GridSearchCV içerisindeki CV splitter
    cv = grid.cv  

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        y_train = y[train_idx]
        y_test  = y[test_idx]

        # Train pozitif/negatif
        train_pos = int(np.sum(y_train == 1))
        train_neg = int(np.sum(y_train == 0))

        # Test pozitif/negatif
        test_pos = int(np.sum(y_test == 1))
        test_neg = int(np.sum(y_test == 0))

        fold_records.append({
            "Fold": fold_idx,
            "Train_Pos(1)": train_pos,
            "Train_Neg(0)": train_neg,
            "Test_Pos(1)": test_pos,
            "Test_Neg(0)": test_neg
        })

    df = pd.DataFrame(fold_records)

    # Excel output file
    filename = os.path.join(
        save_path, 
        f"{rep_name}_{classifier_name}_cv_fold_class_counts.xlsx"#smote_
    )

    df.to_excel(filename, index=False)
    print(f"Fold class counts saved → {filename}")

    return df
    
class SMOTE_Double_Minority(SMOTE):
    def fit_resample(self, X, y):
        # minority sınıf indeksi
        minority_class = 1  # senin etiketin 1 ise
        minority_count = sum(y == minority_class)
        target_count = int(minority_count * 2)  # 2 katına çıkar
        #print(target_count)
        # SMOTE sampling_strategy sözlüğü oluştur
        self.sampling_strategy = {minority_class: target_count}
        
        return super().fit_resample(X, y)


def compute_fmax_from_prob(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_t = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_f1, best_t
    
def print_fold_class_distribution(X, y, cv, random_state=42):
    

    print("\n===== FOLD CLASS DISTRIBUTION =====\n")

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        y_train, y_test = y[train_idx], y[test_idx]

        train_pos = int(np.sum(y_train == 1))
        train_neg = int(np.sum(y_train == 0))

        test_pos = int(np.sum(y_test == 1))
        test_neg = int(np.sum(y_test == 0))

        print(f"--- FOLD {fold_id} ---")
        print(f"Train → Pos: {train_pos} | Neg: {train_neg}")
        print(f"Test  → Pos: {test_pos} | Neg: {test_neg}\n")    
def get_fold_predictions_with_best_model(best_params, X, y, cv, protein_dict,save_path,representation_name_concated,classifier_name):
    """
    GridSearchCV sonunda bulunan en iyi hyperparametrelerle modeli yeniden kurar,
    her fold için model tekrar eğitilir ve predict/predict_proba değerleri döner.
    ZERO LEAKAGE.
    protein_dict: {protein_id: feature_vector}
    """
    fold_thresholds=[]
    fold_trues = []
    fold_preds = []
    fold_probs = []
    fold_fmax=[]
    fold_positive_ids = []
    fold_negative_ids = []
    fold_false_positive_ids = []
    fold_false_negative_ids = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # === Pipeline best params ile yeniden kur ===
        pipeline = Pipeline([
            #('smote', SMOTE_Double_Minority(random_state=42)),
            ('scaler', StandardScaler()),
            ('model_classifier', XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                **best_params
            ))
        ])
        #sampler = pipeline.named_steps['smote']
        #X_res, y_res = sampler.fit_resample(X_train, y_train) 
        # === SADECE TRAIN FOLD İLE EĞİT (NO LEAKAGE) ===
        pipeline.fit(X_train, y_train)
        #breakpoint()
        # === Test fold tahmini ===
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fold_trues.append(y_test)
        fold_preds.append(y_pred)
        fold_probs.append(y_prob)

        # === F-max ve threshold ===
        fmax, best_t = compute_fmax_from_prob(y_test, y_prob)
        fold_fmax.append(fmax)
        fold_thresholds.append(best_t)

        # === Protein ID listelerini oluştur ===
        pos_ids = []
        neg_ids = []
        false_pos_ids = []
        false_neg_ids = []

        for i, idx in enumerate(test_idx):
            # Protein ID'yi feature vector eşlemesi ile bul
            vec = X_test[i]
            protein_id = None
            for pid, vector in protein_dict.items():
                if np.allclose(vector, vec):
                    protein_id = pid
                    break

            if protein_id is not None:
                if y_test[i] == 1:
                    pos_ids.append(protein_id)
                    if y_pred[i] != 1:
                        false_neg_ids.append(protein_id)
                else:
                    neg_ids.append(protein_id)
                    if y_pred[i] != 0:
                        false_pos_ids.append(protein_id)

        fold_positive_ids.append(pos_ids)
        fold_negative_ids.append(neg_ids)
        fold_false_positive_ids.append(false_pos_ids)
        fold_false_negative_ids.append(false_neg_ids)
    
   
    print_fold_class_distribution(X=X_train, y=y_train, cv=cv)
    all_model=pipeline.fit(X, y)
    
    #breakpoint()
    path_test = os.path.join(os.getcwd(),"case_study/case_study_results/test")
   
    filename = os.path.join(path_test, f"{representation_name_concated}_{classifier_name}_model.joblib")
    joblib.dump(all_model, filename)
    return (fold_trues, fold_preds, fold_probs, fold_fmax, fold_thresholds,
            fold_positive_ids, fold_negative_ids,
            fold_false_positive_ids, fold_false_negative_ids)


def select_best_model_with_hyperparameter_tuning(
    representation_name,
    integrated_dataframe,
    scoring_key,
    models=[
    "RandomForestClassifier",
    "SVC",
    "KNeighborsClassifier",
    "XGBoost",
    "Fully_Connected_Neural_ Network",
],

):

    fmax_scorer = make_scorer(fmax_scorer_func)
    
    scoring = {
    "f1_micro": "f1_micro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "auc": "roc_auc",
    "mcc": make_scorer(matthews_corrcoef),
    "fmax": fmax_scorer,
    "sensitivity": sensitivity_scorer,
    "specificity": specificity_scorer
}

    class_len = len(models)
    #import pdb
    #pdb.set_trace()
    
    model_label = np.array(integrated_dataframe["Label"])
    # label_list = [ast.literal_eval(label) for label in integrated_dataframe['Label']]
    protein_representation = integrated_dataframe.drop(["Label", "Entry"], axis=1)
    proteins = list(integrated_dataframe["Entry"])
    #breakpoint()
    vectors = list(protein_representation["Vector"])
    protein_and_representation_dictionary = dict(zip(proteins, vectors))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    protein_representation_array = np.array(
        list(protein_representation["Vector"]), dtype=float
    )
    model_label_array = np.array(model_label)
    predictions_list, result_dict, classifier_name_lst = ([] for i in range(3))

    best_parameter_df = pd.DataFrame(
        columns={"representation_name", "classifier_name", "best parameter"}
    )

    index = 0
    model_count = 0
    representation_name_concated = ""
    file_name = "_"
    path = os.path.join(os.getcwd(),"case_study/case_study_results")
    path_train=os.path.join(path,"training")
    path_test=os.path.join(path,"test")
    if "training" not in os.listdir(path):
        os.makedirs(path_train, exist_ok=True)
        os.makedirs(path_test, exist_ok=True)
    file_name = file_name.join(models)
    best_param_list = []
    for classifier in models:
        index += 1
        m = 0
        model_label_pred_lst, label_lst, protein_name = ([] for i in range(3))

        input_size = len(protein_representation_array[0])

        if classifier == "RandomForestClassifier":
            
            random.seed(random_state)
            np.random.seed(random_state)
            classifier_ = RandomForestClassifier(random_state=random_state) 
            classifier_name = type(classifier_).__name__
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier_)]
            )
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            parameters = {
                "model_classifier__n_estimators": RandomForest_Classifier_parameters.n_estimators,
                "model_classifier__max_depth": RandomForest_Classifier_parameters.max_depth,
                "model_classifier__min_samples_leaf":RandomForest_Classifier_parameters.min_samples_leaf ,
            }

        elif classifier == "SVC":
            
            random.seed(random_state)
            np.random.seed(random_state)
            classifier_ = SVC(random_state=random_state)
            classifier_name = type(classifier_).__name__
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier_)]
            )
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            parameters = {
                "model_classifier__C": SVC_Classifier_parameters.C,
                "model_classifier__gamma": SVC_Classifier_parameters.gamma,
                "model_classifier__kernel":SVC_Classifier_parameters.kernel ,
                "model_classifier__max_iter":SVC_Classifier_parameters.max_iter,
            }

        elif classifier == "KNeighborsClassifier":
            
            random.seed(random_state)
            np.random.seed(random_state)
            classifier_ = KNeighborsClassifier()
            classifier_name = type(classifier_).__name__
            up_limit = int(math.sqrt(int(len(model_label_array) / 5)))
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier_)]
            )
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            k_range = list(range(1, up_limit))
            parameters = {
                "model_classifier__n_neighbors": k_range if len(Kneighbors_Classifier_parameters.n_neighbors)==0 else Kneighbors_Classifier_parameters.n_neighbors,
                "model_classifier__weights":  Kneighbors_Classifier_parameters.weights ,
                "model_classifier__algorithm": Kneighbors_Classifier_parameters.algorithm,
                "model_classifier__leaf_size": list(
                    range(1, int(len(model_label_array) / 5)) if len(Kneighbors_Classifier_parameters.leaf_size)==0 else Kneighbors_Classifier_parameters.leaf_size
                ),
                "model_classifier__p":Kneighbors_Classifier_parameters.p ,
            }

        elif classifier == "XGBoost":

            random.seed(random_state)
            np.random.seed(random_state)
            classifier_ = XGBClassifier(random_state=42)
            classifier_name = type(classifier_).__name__
            up_limit = int(math.sqrt(int(len(model_label_array) / 5)))
            
            model_pipeline = Pipeline([
                #('smote', SMOTE_Double_Minority(random_state=42)),
                ('scaler', StandardScaler()),
                ('model_classifier', XGBClassifier(random_state=42))
            ])
         
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # ---- HYPERPARAMETER GRID ----
            
            class XGB_Classifier_parameters:
                n_estimators = [100, 200, 300]
                max_depth = [3, 5, 7]
                learning_rate = [0.001, 0.01, 0.1]
                subsample = [0.7, 0.9, 1.0]
                colsample_bytree = [0.7, 0.9, 1.0]

            
            parameters = {
                "model_classifier__n_estimators": XGB_Classifier_parameters.n_estimators,
                "model_classifier__max_depth": XGB_Classifier_parameters.max_depth,
                "model_classifier__learning_rate": XGB_Classifier_parameters.learning_rate,
                "model_classifier__subsample": XGB_Classifier_parameters.subsample,
                "model_classifier__colsample_bytree": XGB_Classifier_parameters.colsample_bytree,
            }
            
            
  

        if classifier == "Fully_Connected_Neural_Network":
            
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            #kf = stratified_balanced_kfold(model_label, n_splits=5)
            model_count = model_count + 1
            #breakpoint()
            # classifier_name_lst.append("Neural_Network")
            classifier_name = "Fully_Connected_Neural_Network"
            (
                f_max_cv,
                f_max_cv_train,
                f_max_cv_test,
                loss_train,
                loss,
                loss_tr,
                loss_test,
                protein_name_tr,
                model_label_pred_test_lst,
                label_lst_test,
                model_label_pred_lst,
            ) = ([] for i in range(11))
            (
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
            ) = NN(
                kf,
                path_train,
                protein_representation,
                model_label,
                input_size,
                representation_name,
                protein_and_representation_dictionary,
            )
            if hasattr(kf, "split"):
                splits = list(kf.split(protein_representation_array,model_label_array))
            else:
                splits = kf  # zaten list of (train_idx, test_idx)
            # === KFold splitlerini hazırla ===
            #splits = kf#.split(protein_representation)
            
            fold_results = []   # tüm foldların sonuçlarını burada tutacağız
            fold_wrong_sets = []  # kesişim için fold bazlı yanlış protein ID setleri
            
            for fold_idx in range(len(label_lst_test)):
            
                # Test label ve tahminleri al
                y_true = label_lst_test[fold_idx]
                y_pred = model_label_pred_test_lst[fold_idx]
                splits = list(kf.split(protein_representation_array,model_label_array))

                # Test foldundaki protein ID'lerini al
                _, fold_test_indices = splits[fold_idx]
                
                fold_protein_ids = list(np.array(proteins)[fold_test_indices])
            
                # Test foldunda kaç tane "1" var?
                ones_count = int((y_true == 1).sum())
            
                # Yanlış tahmin edilen indexler
                false_pos_idx = np.where((y_true == 0) & (y_pred == 1))[0]
                false_neg_idx = np.where((y_true == 1) & (y_pred == 0))[0]
            
                false_pos_proteins = [fold_protein_ids[i] for i in false_pos_idx]
                false_neg_proteins = [fold_protein_ids[i] for i in false_neg_idx]
            
                # Fold bazlı yanlış protein seti (kesişim için)
                fold_wrong_sets.append(set(false_pos_proteins + false_neg_proteins))
            
                # DataFrame'e satır olarak ekle
                for pid in false_pos_proteins:
                    fold_results.append({
                        "fold": fold_idx,
                        "test_fmax": f_max_cv_test[fold_idx],
                        "test_ones": ones_count,
                        "protein_id": pid,
                        "true_label": 0,
                        "pred_label": 1,
                        "error_type": "False Positive"
                    })
            
                for pid in false_neg_proteins:
                    fold_results.append({
                        "fold": fold_idx,
                        "test_fmax": f_max_cv_test[fold_idx],
                        "test_ones": ones_count,
                        "protein_id": pid,
                        "true_label": 1,
                        "pred_label": 0,
                        "error_type": "False Negative"
                    })
            
            # === Son DataFrame (fold bazlı hata listesi) ===
            df_fold_errors = pd.DataFrame(fold_results)
            
            # Kaydet
            error_file_path = os.path.join(
                path,
                "test",
                f"{representation_name[0]}_{classifier_name}_all_fold_test_errors.xlsx" 
            )
            
            df_fold_errors.to_excel(error_file_path, index=False)
            print("Fold-based test error report saved:", error_file_path)
            
            
            # ===============================================================
            # === 5 FOLD ORTAK YANLIŞ TAHMİN EDİLEN PROTEİN KESİŞİMİ =========
            # ===============================================================
            
            """if len(fold_wrong_sets) > 0:
                intersection_wrong = set.intersection(*fold_wrong_sets)
            else:
                intersection_wrong = set()
            
            print("\nIntersection of wrong predictions across all 5 folds:")
            print(intersection_wrong)
            
            # Tek satırlık DataFrame
            df_intersection = pd.DataFrame({
                "wrong_in_all_folds": [list(intersection_wrong)]
            })
            
            intersection_file_path = os.path.join(
                path,
                "test",
                f"{representation_name[0]}_{classifier_name}_wrong_intersection_10.xlsx"
            )
            
            df_intersection.to_excel(intersection_file_path, index=False)
            
            print("Saved intersection error file:", intersection_file_path)"""



            #breakpoint()
            
            neural_network_eval(
                f_max_cv_train,
                kf,
                model,
                model_label_pred_lst,
                label_lst,
                index,
                representation_name,
                classifier_name,
                file_name,
                "training",
                protein_name_tr,
                path,
                parameter,
            )

            neural_network_eval(
                f_max_cv_test,
                kf,
                model,
                model_label_pred_test_lst,
                label_lst_test,
                index,
                representation_name,
                classifier_name,
                file_name,
                "test",
                protein_name,
                path,
                parameter,
            )
                # ===== TRAIN 1/0 COUNTS =====
            train_pos_list = []
            train_neg_list = []
            for fold_idx in range(len(label_lst)):
                #breakpoint()
                y_tr = label_lst[fold_idx]
                train_pos_list.append(int((y_tr == 1).sum()))
                train_neg_list.append(int((y_tr == 0).sum()))
            
            # ===== TEST 1/0 COUNTS =====
            test_pos_list = []
            test_neg_list = []
            for fold_idx in range(len(label_lst_test)):
                y_te = label_lst_test[fold_idx]
                test_pos_list.append(int((y_te == 1).sum()))
                test_neg_list.append(int((y_te == 0).sum()))
            
            # ===== EXCEL KAYDI =====
            save_fold_metrics_xlsx(
                save_path=path_test,
                representation_name=representation_name,
                classifier_name=classifier_name,
                fmax_list=f_max_cv_test,
                train_pos_list=train_pos_list,
                train_neg_list=train_neg_list,
                test_pos_list=test_pos_list,
                test_neg_list=test_neg_list,
                eval_type="test"
            )
            
            save_fold_metrics_xlsx(
                save_path=path_train,
                representation_name=representation_name,
                classifier_name=classifier_name,
                fmax_list=f_max_cv_train,
                train_pos_list=train_pos_list,
                train_neg_list=train_neg_list,
                test_pos_list=test_pos_list,
                test_neg_list=test_neg_list,
                eval_type="training"
            )

        else:
            model_count = model_count + 1
        
            if scoring_key[0] == "f_max":        
                model_tunning = GridSearchCV(
                    estimator=model_pipeline,
                    param_grid=parameters,
                    cv=kf,
                    pre_dispatch=20,
                    scoring=F_max_scoring.scoring_f_max_machine,
                    n_jobs=-1
                )
            else:
                model_tunning = GridSearchCV(
                    estimator=model_pipeline,
                    param_grid=parameters,
                    cv=kf,
                    pre_dispatch=20,
                    scoring=scoring_function_dictionary[scoring_key[0]],
                    n_jobs=-1
                )   
        
            classifier_name_lst.append(classifier_name)
            model_tunning.fit(protein_representation_array, model_label)
        
            # best_params al
            best_params = model_tunning.best_params_
        
            representation_name_concated = "_".join(representation_name)
            best_parameter_df = best_parameter_df.append(
                {
                    "representation_name": representation_name_concated + "_binary_classifier",
                    "classifier_name": classifier_name,
                    "best parameter": best_params,
                },
                ignore_index=True,
            )
            best_param_list.append(
                {
                    "representation_name": representation_name_concated + "_binary_classifier",
                    "classifier_name": classifier_name,
                    "best parameter": best_params,
                }
            )
            
            # === Her fold için pipeline oluşturup fit et ===
            (fold_trues, fold_preds, fold_probs, fold_fmax, fold_thresholds,
                 fold_positive_ids, fold_negative_ids,
                 fold_false_positive_ids, fold_false_negative_ids) = get_fold_predictions_with_best_model(
                    best_params=best_params,
                    X=protein_representation_array,
                    y=model_label,
                    cv=kf,
                    protein_dict=protein_and_representation_dictionary,save_path=path,representation_name_concated=representation_name_concated,classifier_name=classifier_name

                )
            rows = []
            for fold_id in range(len(fold_trues)):
                row = {
                    "Fold": fold_id,
                    "F_max": fold_fmax[fold_id],
                    "Threshold": fold_thresholds[fold_id],
                    "Train_Pos(1)": len(np.where(model_label[kf.split(protein_representation_array, model_label).__next__()[0]] == 1)[0]),
                    "Train_Neg(0)": len(np.where(model_label[kf.split(protein_representation_array, model_label).__next__()[0]] == 0)[0]),
                    "Test_Pos(1)": len(fold_positive_ids[fold_id]),
                    "Test_Neg(0)": len(fold_negative_ids[fold_id]),
                    "False_Pos_IDs": ",".join([str(i) for i in fold_false_positive_ids[fold_id]]),
                    "False_Neg_IDs": ",".join([str(i) for i in fold_false_negative_ids[fold_id]]),
                    "Pos_IDs": ",".join([str(i) for i in fold_positive_ids[fold_id]]),
                    "Neg_IDs": ",".join([str(i) for i in fold_negative_ids[fold_id]]),
                }
                rows.append(row)
            
            df_fold = pd.DataFrame(rows)
            
            # CSV kaydet
            
            os.makedirs(path_test, exist_ok=True)
            df_fold.to_csv(os.path.join(path_test, representation_name_concated + "_fold_summary.csv"), index=False)
            
            #print(f"Fold summary kaydedildi: {os.path.join(path,"training", "_representation_name_concated" + "_fold_summary.csv")}")
                    
            # === Fold bazlı değerlendirme ===
            binary_evaluate.evaluate(
                kf,
                fold_preds,
                fold_trues,
                fold_fmax,
                classifier_name,
                representation_name_concated,
                file_name,
                index,
                "test"
            )
        
            # === Prediction CSV ===
            protein_name = []  # protein_id listesini fold_trues/fold_preds üzerinden çıkarabilirsin
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(protein_representation_array, model_label)):
                for vec in protein_representation_array[test_idx]:
                    for protein, vector in protein_and_representation_dictionary.items():
                        if str(vector) == str(list(vec)):
                            protein_name.append(protein)
                            break
        
            label_predictions = pd.DataFrame(
                np.concatenate(fold_preds), columns=["Label"]
            )
            label_predictions.insert(0, "protein_id", protein_name)
            label_predictions.to_csv(
                os.path.join(path_test, representation_name_concated + "_binary_classifier_test_predictions.csv"),
                index=False
            )
        
            class_name = "_".join(classifier_name_lst)
            best_parameter_df.to_csv(
                os.path.join(path_test, representation_name_concated + "_" + class_name + "_binary_classifier_best_parameter.csv"),
                index=False
            )

    return best_param_list

