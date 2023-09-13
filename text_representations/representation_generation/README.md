# textrep

This repository contains Python scripts to generate text representations using different techniques. It provides options to create TF-IDF representations, BioSentVec representations, and BioWordVec representations for input data.  It takes input data from UniProt and PubMed sources and generates vector representations for each entry. 

Dataset is temporarily limited to 20 entries to make testing easier. PCA analysis for tfidf vectors also disabled because the limit is less than the PCA components.

## Definition of Scripts

### create_tfidf.py

#### Functions

`create_reps(tp)`: This function takes a parameter tp which represents the type of data ("uniprot", "pubmed", or "uniprotpubmed"). It reads files from the specified paths (ufiles_path and pfiles_path), concatenates their contents based on the data type, and applies the TF-IDF vectorization technique using the TfidfVectorizer from scikit-learn. It creates a pandas DataFrame (df1) where each row represents a file, and each column represents a feature (word). The DataFrame is then saved as a CSV file. It also performs Principal Component Analysis (PCA) on the DataFrame using different numbers of components (256, 512, 1024, 2048), and saves the resulting PCA-transformed vectors as separate CSV files.


### create_biosentvec.py

#### Functions

`convert_dataframe_to_multi_col(representation_dataframe)`: This function takes a representation DataFrame as input (representation_dataframe), which has two columns: 'Entry' and 'Vector'. It splits the 'Vector' column into separate columns and merges them with the 'Entry' column to create a new DataFrame with multiple columns for each dimension of the vector. The resulting DataFrame with multiple columns is returned.

`preprocess_sentence(text)`: This function takes a text string (text) as input and performs preprocessing on it. It replaces specific characters, converts the text to lowercase, tokenizes the text using NLTK's word_tokenize function, removes stopwords and punctuation, and returns the preprocessed text as a string.

`create_reps(tp)`: This function takes a parameter tp which represents the type of data ("uniprot", "pubmed", or "uniprotpubmed"). It loads a pre-trained sentence embedding model using the sent2vec library. It then iterates over files in the specified paths (ufiles_path and pfiles_path), reads the content of each file, preprocesses the content into a sentence, and obtains the sentence embedding using the loaded model. The entry name and sentence embedding are stored in a DataFrame (df). The DataFrame is then converted to a multi-column representation using the convert_dataframe_to_multi_col function. Finally, the resulting DataFrame is saved as a CSV file.


### create_biowordvec.py

#### Functions

`convert_dataframe_to_multi_col(representation_dataframe)`: This function takes a representation DataFrame as input (representation_dataframe), which has two columns: 'Entry' and 'Vector'. It splits the 'Vector' column into separate columns and merges them with the 'Entry' column to create a new DataFrame with multiple columns for each dimension of the vector. The resulting DataFrame with multiple columns is returned.

`preprocess_sentence(text)`: This function takes a text string (text) as input and performs preprocessing on it. It replaces specific characters, converts the text to lowercase, tokenizes the text using NLTK's word_tokenize function, removes stopwords and punctuation, and returns the preprocessed text as a string.

`create_reps(tp)`: This function takes a parameter tp which represents the type of data ("uniprot", "pubmed", or "uniprotpubmed"). It loads a pre-trained word embedding model using the fasttext library. It then iterates over files in the specified paths (ufiles_path and pfiles_path), reads the content of each file, preprocesses the content into a sentence, and obtains the sentence embedding using the loaded model. The entry name and sentence embedding are stored in a DataFrame (df). The DataFrame is then converted to a multi-column representation using the convert_dataframe_to_multi_col function. Finally, the resulting DataFrame is saved as a CSV file.

### createtextrep.py

#### Functions

This script provides a convenient way to create different types of text representations by specifying the desired representation techniques and the paths to the input files via command-line arguments.

Creating TFIDF representations: If the tfidf or all flags are set, the script sets the file paths for the create_tfidf module and calls its main() function to create TFIDF representations.

Creating biosentvec representations: If the biosentvec or all flags are set, the script sets the file paths for the create_biosentvec module and calls its main() function to create biosentvec representations.

Creating biowordvec representations: If the biowordvec or all flags are set, the script sets the file paths for the create_biowordvec module and calls its main() function to create biowordvec representations.

## Data

Uniprot and Pubmed files must be in text format and named with the uniprot ids. Download and unzip the files to the data folder from the urls given below.

https://drive.google.com/file/d/1jZJiL6R9c4hsxh_k5pCBsX6LG1zzbITX/view?usp=drive_link
https://drive.google.com/file/d/1BwU2DXCXdtHGxtY1TlQxTuNbc7xVBzDp/view?usp=drive_link

## Models

biosentvec and biowordvec models must be downloaded to models folder from the urls below.

https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin
https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin


### Options

The script allows users to specify different options to create specific types of text representations (TFIDF, biosentvec, and biowordvec) and provides flexibility by allowing the creation of all representation types if the -a or --all option is specified.

`-tfidf` or `--tfidf`: Creates TFIDF representations.

`-bsv` or `--biosentvec`: Creates biosentvec representations.

`-bwv` or `--biowordvec`: Creates biowordvec representations.

`-upfp` or `--uniprotfilespath`: Specifies the path for the uniprot files. This option is required.

`-pmfp` or `--pubmedfilespath`: Specifies the path for the pubmed files. This option is required.

`-a` or `--all`: Creates all types of representations (TFIDF, biosentvec, and biowordvec).

### How to Run

Step by step operation:
  1. Clone repository
  2. Install dependencies(given below)
  3. Download biosentvec and biowordvec models to models folder
  4. Download and unzip uniprot and pubmed files to data folder
  5. Run the script

Examples:

1. To create TF-IDF representations:

```
python createtextrep.py --tfidf -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
```

2. To create biosentvec representations:

```
python createtextrep.py --bsv -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
```

2. To create biowordvec representations:

```
python createtextrep.py --bwv -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
```

3. To create all three representations:

```
python createtextrep.py --a -upfp /path/to/uniprot/files -pmfp /path/to/pubmed/files
```

# Dependencies
 1.	Python 3.7.3
 2.	pandas 1.1.4
 3.	sklearn
 4.	os
 5.	fasttext
 6.	string
 7.	nltk
 8.	sent2vec

## Definition of Output

The script will load the text files and perform the selected actions based on the provided options. The output will be generated in the following manner:

- If the `-tfidf` option is selected, a csv file inluding TF-IDF vectors and four csv files (PCA 256, PCA 512, PCA 1024 and PCA 2048) including vectors generated by PCA analysis will be created and saved.

- If the `-bsv` option is selected, a csv file including biosentvec vectors will be created and saved.

- If the `-bwv` option is selected, a csv file including biowordvec vectors will be created and saved.

## License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
