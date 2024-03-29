# Protein Function Prediction

- Protein Function Prediction aims to construct models for protein function prediction. It can concate protein representations in to prepare datasets  for training, testing models and making predictions.
- We construct a model consisting of 4 steps that can be used independently or contiguously.
- We compare it with  other methods from the literature.
  
 **1. Fuse_representations:**
 - This step make concatenation of protein representation vectors.
   
 **2. Prepare_datasets:**
  - Concation of positive_sample_dataset and negative_sample_dataset for preparation of model dataset ( "Entry" and multi-columns representation vector )
  - Save pickle format of dataset
    
 **3. Model_training and Test:**
  - Training and test for prepared data. Using models are Fully Connected Neural Network, RandomForestClassifier, SVC, KNeighborsClassifier
    
  **4. Model_prediction:** 
  - Make prediction for protein function
You can access the information about the fuse_representations, prepare_datasets, model_training_test, prediction sub-modules of the function prediction module via the .py extension file of the relevant sub-module.

# How to run Protein Function Prediction

Step by step operation:
  1. Clone repository
  3. Download datasets,unzip and place the folder
  4. Edit the configuration file : **Hoper.yaml** 
- i.e., python **HOPER_main.py**

# Dependencies

 You can access dependencies from case_study_env.yml file
 
# Example of protein function prediction configuration file 

    parameters:
    choice_of_task_name:  [prepare_datasets,model_training_test,prediction]
    fuse_representations: ** # This step make concatenation of protein representation vectors.**
        representation_files: [../multi_modal_rep_ae_multi_col_256.csv,/media/DATA2/sinem/node2vec_d_50_p_0.5_q_0.25_multi_col.csv]
        min_fold_number:  2  # Minimum_number_of_combinations. For example if 3 representations are supplied and min_fold_number = 2
        #then the function will produce double and triple combinations of the protein representation vector. Such as Vec1_Vec2, Vec1_Vec3, Vec2_Vec3 and Vec1_Vec2_Vec3
        representation_names:  [modal_rep_ae,node2vec,bertavg]     
        
    prepare_datasets:  
        positive_sample_data:  [../positive.csv]
        negative_sample_data:  [../neg_data.csv]
        prepared_representation_file:  [../multi_modal_rep_ae_multi_col_256.csv] 
        representation_names:  [modal_rep_ae] 
    
    model_training_test:
        representation_names:  [modal_rep_ae]
        scoring_function:  ["f_max"]  # "f1_micro","f1_macro", "f1_weighted", "f_max"
        prepared_path:  ["../results/modal_rep_ae_binary_data.pickle"]
        classifier_name:  ["Fully_Connected_Neural_Network"] #"RandomForestClassifier", "SVC", "KNeighborsClassifier", "Fully_Connected_Neural_ Network",
    prediction:
        representation_names:  [modal_rep_ae] 
        prepared_path:  ["../rep_file/rep_dif_ae.csv"]
        classifier_name:  ['Fully_Connected_Neural_Network']         
        model_directory:  ["../case_study_results/test/modal_rep_ae_Fully_Connected_Neural_Network_binary_classifier.pt"] 

# Definition of output files (results)

"representation_names_fused_representations_dataframe_multi_col.csv": Prepared dataset

- Training result files:

   - "case_study_results/training/representation_names_model_name_training.tsv": training results which contains 29 columns

| Column names | Column names | Column names |
| ------------- | ------------- | ------------- |
|  "representation_name"  | "classifier_name"  | "accuracy" |
|  "std_accuracy"  | "f1_micro"  |  "std_f1_micro" |
|   "f1_macro"  | "std_f1_macro"  | "f_max" |
|  "std_f_max"  | "f1_weighted"  | "std_f1_weighted" |
|  "precision_micro"  | "std_precision_micro"  | "precision_macro" |
|  "std_precision_macro"  | "precision_weighted" |  "std_precision_weighted" |
|  "recall_micro" | "std_recall_micro"  | "recall_macro" |
|   "std_recall_macro"  | "recall_weighted"  | "std_recall_weighted" |
| "hamming distance"  | "std_hamming distance"  | "auc" |
|  "std_auc"  | "matthews correlation coefficient" |  |
   
  
    - "case_study_results/training/representation_model_name_binary_classifier.pt" : saved model

    - "case_study_results/training/representation_model_name_means.tsv" : mean of 5 fold results

    - "case_study_results/training/model_name_representation_name_binary_classifier_best_parameter.csv"

    - "case_study_results/training/representation_name_model_name_binary_classifier_training_predictions.tsv"


- Test result files:

   - "case_study_results/test/representation_names_model_name_test.tsv":contains same columns as training results

    - "case_study_results/test/representation_model_name_binary_classifier.pt" : saved model

    - "case_study_results/test/representation_name_model_name_test_means.tsv" : mean of 5 fold results

    - "case_study_results/test/model_name_representation_name_binary_classifier_best_parameter.csv"

    - "case_study_results/test/representation_name_model_name_binary_classifier_test_predictions.tsv"
 - Prediction result files:
   - "case_study_results/prediction/Representation_name_prediction_binary_classifier_classifier_name.tsv"

