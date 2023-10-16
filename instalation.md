# HOPER Instalation Steps For Models

## PPI Model Instalation Instructions
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

## text Model Installation Instructions

* To use text representation generator, copy uniprot and pubmed text files to HOPER/text_representations/representation_generation/data/
* biosentvec and biowordvec models must be downloaded to HOPER/text_representations/representation_generation/models from the urls given below. Alternatively model_download parameter must be set as "y" to download models automatically if biosentvec or biowordvec representations selected to be generated.

https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin
https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
