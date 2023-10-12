## HOPER Instalation Steps For Models

# PPI Model Instalation Instructions
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
# Text Representation Models Instalation Instructions

Download biosentvec and biowordvec models to models folder
Download and unzip uniprot and pubmed files to data folder
Run the script
Examples:

To create TF-IDF representations:
python createtextrep.py --tfidf -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
To create biosentvec representations:
python createtextrep.py --bsv -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
To create biowordvec representations:
python createtextrep.py --bwv -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
To create all three representations:
python createtextrep.py --a -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
