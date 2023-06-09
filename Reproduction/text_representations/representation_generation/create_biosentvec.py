import os
import pandas as pd
import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm

ufiles_path = ''
pfiles_path = ''
stop_words = set(stopwords.words('english'))

def convert_dataframe_to_multi_col(representation_dataframe):
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry,right=vector,left_index=True, right_index=True)
    return multi_col_representation_vector
    
def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()
    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
    return ' '.join(tokens)

def create_reps(tp):
    files = os.listdir(pfiles_path)
    model = sent2vec.Sent2vecModel()
    print("\n\nLoading model...\n")
    try:
            model.load_model('path_to_biosentvec_model')
    except Exception as e:
            print(e)
            print('model successfully loaded')

    df = pd.DataFrame(columns=['Entry', 'Vector'])
    print("\n\nCreating " + tp + "biosentvec vectors...\n")
    for i in tqdm(range(len(files))):
        if tp == 'uniprot':
            contentu = open(ufiles_path + files[i])
            sentence = preprocess_sentence(contentu.read())
        elif tp == 'pubmed':
            contentp = open(pfiles_path + files[i])
            sentence = preprocess_sentence(contentp.read())
        elif tp == 'uniprotpubmed':
            contentu = open(ufiles_path + files[i])
            contentp = open(pfiles_path + files[i])
            sentence = preprocess_sentence(contentu.read() + contentp.read())
        
        sentence_vector = model.embed_sentence(sentence)
        df = df.append({'Entry' : files[i][:-4], 'Vector' : sentence_vector[0]}, ignore_index = True)

    df = convert_dataframe_to_multi_col(df)
    df.to_csv('path_to_save_file/' + tp + '_biosentvec_vectors_multi_col.csv', index = False)

def main():
    create_reps("uniprot")
    create_reps("pubmed")
    create_reps("uniprotpubmed")
