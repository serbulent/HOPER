import os
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm
from openai import OpenAI

ufiles_path = ''
pfiles_path = ''
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
create_reps(tp): This function takes a parameter tp which represents the type of data ("uniprot", "pubmed", or "uniprotpubmed"). It loads a pre-trained word embedding model. 
It then iterates over files in the specified paths (ufiles_path and pfiles_path), reads the content of each file, preprocesses the content into a sentence, and obtains the sentence embedding using the loaded model. 
The entry name and sentence embedding are stored in a DataFrame (df). The DataFrame is then converted to a multi-column representation using the convert_dataframe_to_multi_col function. 
Finally, the resulting DataFrame is saved as a CSV file.
'''

def create_reps(tp):
    path=os.getcwd()
    files = os.listdir(pfiles_path)
    print("\n\nLoading model...\n")
    client = OpenAI(
        api_key="your OpenAI api key"
    )

    data = []
    
    print("\n\nCreating " + tp + " openai embeddings...\n")
    #for i in tqdm(range(20)):
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
        
        try:
            response = client.embeddings.create(
                input=sentence[:8191],
                model="text-embedding-3-large"
            )
            
            sentence_vector = response.data[0].embedding
            data.append([ files[i][:-4], sentence_vector])
            
        except Exception as e:
            print("Error during API call:", e)
            continue  # or handle accordingly

    df = pd.DataFrame(data, columns=['Entry', 'Vector']) 
    df = convert_dataframe_to_multi_col(df)
    df.to_csv(os.path.join(path,'text_representations/representation_generation/openai_representations/' + tp + '_openai_large_vectors_multi_col.csv'), index = False)

def main():
    create_reps("uniprot")
    create_reps("pubmed")
    create_reps("uniprotpubmed")