
# HoloprotRep: Protein Representation Binary Classification

- HoloprotRep aims to construct models for protein function prediction Holoprotrep can concate protein representations in to prepare datasets  for training and testing models
We construct a model consisting of  4 steps that can be used independently or contiguously:
We compare it with  other methods from the literature.
 1. Fuse_representations:
    - Concatenate of protein representation vectors
 2. Prepare_datasets:
    - Concantation of positive_sample_dataset and negative_sample_dataset for preperation of dataset which have "Entry" and multi columns representation vector and  save pickle format of dataset   

 3. Model_training:
    - Training and test for prepared data. Using models are Fully Connected Neural Network,RandomForestClassifier,SVC,KNeighborsClassifier
 6. Model_prediction:
    - Make prediction for binary label

# Example of binary classification configuration file 


Parameters:

    ##choice of task, task names can be fuse_representations,prepare_datasets,model_training,model_test
    
    choice_of_task_name:  [model_training]
    
    ##protein representation model name for result files
    
    representation_names: [modal_rep_ae]
    ##representation vectors concantenation
    fuse_representations:
    
        representation_files: [../data/HoloProtReti_modal_rep_ae_multi_col_256.csv]
    ## Concantation of positive_sample_dataset and negative_sample_dataset for preperation of dataset which have "Entry" and multi columns representation vector and  save pickle format of dataset   
    prepare_datasets:  
    
        positive_sample_data:  [../data/positive.csv]
        negative_sample_data:  [../data/sinem/neg_data.csv]
        prepared_representation_file:  [../data/multi_modal_rep_ae_multi_col_256.csv] 
        
    ## Training of prepared data
    model_training:
    
        auto:  True
        ## if run training alone enter prepared path
        prepared_path:  [../data/results/modal_rep_ae_binary_data.pickle]
        ## Enter classifier name which can be "Fully Connected Neural Network","RandomForestClassifier","SVC","KNeighborsClassifier"
        classifier_name:  ["Fully Connected Neural Network"] 
        
    prediction:
        ## if run training alone enter prepared path
        prepared_path:  ["../data/rep_dif_ae.csv"]
        classifier_name:  ['Fully Connected Neural Network']         
        model_directory:  ["../results/test/modal_rep_ae_binary_classifier_Fully Connected Neural Network.pt"] 
        
# Definition of output files (results)

"representation_names_fused_representations_dataframe_multi_col.csv": Prepared dataset

- Training result files:

   - "training/representation_names_model_name_training.tsv": training results which contains "representation_name","classifier_name", "accuracy", "std_accuracy",   "f1_micro",   "std_f1_micro", "f1_macro", "std_f1_macro","f_max", "std_f_max", "f1_weighted","std_f1_weighted","precision_micro", "std_precision_micro","precision_macro",    "std_precision_macro", "precision_weighted", "std_precision_weighted", "recall_micro", "std_recall_micro","recall_macro", "std_recall_macro", "recall_weighted",     "std_recall_weighted",  "hamming distance","std_hamming distance","auc","std_auc","matthews correlation coefficient" columns

    - "training/representation_model_name_binary_classifier.pt" : saved model

    - "training/representation_model_name_means.tsv" : mean of 5 fold results

    - "training/model_name_representation_name_binary_classifier_best_parameter.csv"

    - "training/representation_name_model_name_binary_classifier_training_predictions.tsv"


- Test result files:

   - "test/representation_names_model_name_test.tsv": training results which contains "representation_name","classifier_name", "accuracy", "std_accuracy",   "f1_micro",   "std_f1_micro", "f1_macro", "std_f1_macro","f_max", "std_f_max", "f1_weighted","std_f1_weighted","precision_micro", "std_precision_micro","precision_macro",    "std_precision_macro", "precision_weighted", "std_precision_weighted", "recall_micro", "std_recall_micro","recall_macro", "std_recall_macro", "recall_weighted",     "std_recall_weighted",  "hamming distance","std_hamming distance","auc","std_auc","matthews correlation coefficient" columns

    - "test/representation_model_name_binary_classifier.pt" : saved model

    - "test/representation_name_model_name_test_means.tsv" : mean of 5 fold results

    - "test/model_name_representation_name_binary_classifier_best_parameter.csv"

    - "test/representation_name_model_name_binary_classifier_test_predictions.tsv"
 - Prediction result files:
   - "Representation_name_prediction_binary_classifier_classifier_name.tsv"
