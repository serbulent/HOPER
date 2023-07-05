# HOPER (Holistic Protein Representation)

-Holistic protein representation uses  multimodal learning model to predict protein functions even with low-data. 

-Representation vectors created using protein sequence, protein text and protein-protein interaction data types to achieve this goal.

-The rationale behind  incorporating protein-protein interactions into our holistic protein representation model is the assumption 
that interacting proteins are likely to act in the same biological process. Also, these proteins are probably located at the same location in the cell. 

-Text-based protein representations calculated with pre-trained natural language processing models.

-We aim to increase low-data prediction performance by using these three data types together.

# How to run HOPER

Step by step operation:
  1. Clone repository: git clone https://github.com/serbulent/HOPER.git
  2. Install dependencies(given below): i.e, pip install pandas==1.1.4 etc.
  3. Download datasets,unzip and place the your folder
  4. Edit the configuration file (Necessary adjustments (for example, giving the file location) are made in the configuration file of the method intended to be used.
How the editing should be done is in the README.md file of the relevant module, for case study make edition on **Protein_Function_Prediction_config.yaml** configuration file
  5. Run module main function  i.e., python **Function_prediction_main.py**

# Dependencies
 1.	Python 3.7.3
 2.	pandas 1.1.4
 3.	scikit-learn 0.22.1.
 4.	Scikit-MultiLearn

- Example of case study configuration file see documentation [https://github.com/serbulent/HOPER/Reproduction/case_study/readme.md](readme.md)
