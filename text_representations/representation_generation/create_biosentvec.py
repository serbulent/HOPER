import os
import pandas as pd
import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm

import rep_paths

ufiles_path = ''
pfiles_path = ''
rep_paths.ensure_nltk_data()
stop_words = set(stopwords.words('english'))

'''
convert_dataframe_to_multi_col(representation_dataframe): This function takes a representation DataFrame as input (representation_dataframe), which has two columns: 'Entry' and 'Vector'. 
It splits the 'Vector' column into separate columns and merges them with the 'Entry' column to create a new DataFrame with multiple columns for each dimension of the vector. 
The resulting DataFrame with multiple columns is returned.
'''

def convert_dataframe_to_multi_col(representation_dataframe):
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry,right=vector,left_index=True, right_index=True)
    return multi_col_representation_vector

'''
preprocess_sentence(text): This function takes a text string (text) as input and performs preprocessing on it. It replaces specific characters, converts the text to lowercase, 
tokenizes the text using NLTK's word_tokenize function, removes stopwords and punctuation, and returns the preprocessed text as a string.
'''

def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()
    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
    return ' '.join(tokens)

'''
create_reps(tp): This function takes a parameter tp which represents the type of data ("uniprot", "pubmed", or "uniprotpubmed"). It loads a pre-trained sentence embedding model using the sent2vec library. 
It then iterates over files in the specified paths (ufiles_path and pfiles_path), reads the content of each file, preprocesses the content into a sentence, and obtains the sentence embedding using the loaded model. 
The entry name and sentence embedding are stored in a DataFrame (df). The DataFrame is then converted to a multi-column representation using the convert_dataframe_to_multi_col function. 
Finally, the resulting DataFrame is saved as a CSV file.
'''

def create_reps(tp):
    files = sorted(os.listdir(pfiles_path))
    if not hasattr(sent2vec, "Sent2vecModel"):
        raise SystemExit("The installed 'sent2vec' package is not epfml/sent2vec. Re-run create_env.sh.")
    rep_paths.require_model(rep_paths.BIOSENTVEC_MODEL, "BioSentVec")
    model = sent2vec.Sent2vecModel()
    print("\n\nLoading model...\n")
    model.load_model(rep_paths.BIOSENTVEC_MODEL)

    rows = []
    print("\n\nCreating " + tp + " biosentvec vectors...\n")
    for i in tqdm(range(len(files))):
        if tp == 'uniprot':
            sentence = preprocess_sentence(rep_paths.read_text(ufiles_path, files[i]))
        elif tp == 'pubmed':
            sentence = preprocess_sentence(rep_paths.read_text(pfiles_path, files[i]))
        elif tp == 'uniprotpubmed':
            sentence = preprocess_sentence(rep_paths.read_text(ufiles_path, files[i]) + rep_paths.read_text(pfiles_path, files[i]))

        sentence_vector = model.embed_sentence(sentence)
        rows.append({'Entry': os.path.splitext(files[i])[0], 'Vector': sentence_vector[0]})

    df = convert_dataframe_to_multi_col(pd.DataFrame(rows, columns=['Entry', 'Vector']))
    df.to_csv(os.path.join(rep_paths.output_dir('biosentvec'), tp + '_biosentvec_vectors_multi_col.csv'), index = False)

def main():
    create_reps("uniprot")
    create_reps("pubmed")
    create_reps("uniprotpubmed")
