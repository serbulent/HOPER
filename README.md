# Currently Under Improvment! We are diligently streamlining this repository for seamless automation!
# HOPER (Holistic Protein Representation)
![](Figures/figure_.jpg 100*20)


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

* To install packages to use for Node2vec and HOPE in your home directory, use:

  * GEM version 213189b; use for old version:
  
    git clone [https://github.com/palash1992/GEM.git]
    
    git checkout  [213189b]

* To make Node2vec executable; Clone repository git clone https://github.com/snap-stanford/snap and Compiles SNAP. The code for compiles is as below:
  
  - cd snap/
  - rm -rf examples/Release
  - make all
  - cd examples/node2vec
  - chmod +x node2vec
  - ls -alh node2vec

* Make node2vec executable and add to system PATH or move it to the location you run.

- Example of case study configuration file see documentation [readme.md](https://github.com/serbulent/HOPER/blob/main/Reproduction/case_study/readme.md)
