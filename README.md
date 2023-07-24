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
  2. Edit the configuration file Hoper.yaml
  3. Run module main function  i.e., python **HOPER_main.py**

- Example of case study configuration file see documentation [readme.md](https://github.com/serbulent/HOPER/blob/main/Reproduction/case_study/readme.md)
