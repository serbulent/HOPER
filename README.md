# HOPER (Holistic Protein Representation)

-Holistic protein representation uses  multimodal learning model to predict protein functions even with low-data. 
-Representation vectors created using protein sequence, protein text and protein-protein interaction data types to achieve this goal.
-The rationale behind  incorporating protein-protein interactions into our holistic protein representation model is the assumption 
that interacting proteins are likely to act in the same biological process. Also, these proteins are probably located at the same location in the cell.  
-Text-based protein representations calculated with pre-trained natural language processing models.
-We aim to increase low-data prediction performance by using these three data types together.

# How to run HOPER

Step by step operation:
  1. Clone repostory
  2. Install dependencies(given below)
  3. Download datasets,unzip and place the folder
  4. Edit the configuration file (according to classification methods, for binary classification: **file_name_config.yaml** 
- i.e., python **HOPER_binary_label.py**

# Dependencies
 1.	Python 3.7.3
 2.	pandas 1.1.4
 3.	scikit-learn 0.22.1.
 4.	Scikit-MultiLearn

- Example of binary classification configuration file see documentation [binary_classification.md](binary_classification.md)

-Prepraration of the input vector dataset: 
- Generate your representation vectors for all human proteins ([amino acid sequences of canonical isoform human proteins](https://drive.google.com/file/d/1wXF2lmj4ZTahMrl66QpYM2TvHmbcIL6b/view?usp=sharing)), and for the samples in the [SKEMPI dataset](https://drive.google.com/file/d/1m5jssC0RMsiFT_w-Ykh629Pw_An3PInI/view?usp=sharing).
