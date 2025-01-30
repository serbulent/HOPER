import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import gc

ufiles_path = ''
pfiles_path = ''

'''
convert_dataframe_to_multi_col(representation_dataframe): This function takes a representation DataFrame as input (representation_dataframe), which has two columns: 'Entry' and 'Vector'. 
It splits the 'Vector' column into separate columns and merges them with the 'Entry' column to create a new DataFrame with multiple columns for each dimension of the vector. 
The resulting DataFrame with multiple columns is returned.
'''

def convert_dataframe_to_multi_col(representation_dataframe, id_column):
    entry = pd.DataFrame(representation_dataframe[id_column])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry, right=vector, left_index=True, right_index=True)
    return multi_col_representation_vector

def create_reps(tp):
    files = os.listdir(pfiles_path)
    file_list = []

    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    data = []
    print('Generating ' + tp + ' embeddings...')
    for i in tqdm(range(len(files))):
        file_content = ""
        if tp == 'uniprot':
            with open(os.path.join(ufiles_path, files[i])) as contentu:
                file_content = contentu.read()
        elif tp == 'pubmed':
            with open(os.path.join(pfiles_path, files[i])) as contentp:
                file_content = contentp.read()
        elif tp == 'uniprotpubmed':
            with open(os.path.join(ufiles_path, files[i])) as contentu, open(os.path.join(pfiles_path, files[i])) as contentp:
                file_content = contentu.read() + contentp.read()
        
        inputs = tokenizer(file_content, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :] 
            flat_embedding = cls_embedding.squeeze().numpy().flatten()
            data.append([files[i][:-4], flat_embedding])
            
        del cls_embedding, outputs
        gc.collect()

    df = pd.DataFrame(data, columns=['Entry', 'Vector']) 
    df = convert_dataframe_to_multi_col(df, id_column='Entry')
    df.to_csv(os.path.join(path,'text_representations/representation_generation/biobert_representations/' + tp + '_biobert_embeddings_multi_col.csv'), index = False)
   
def main():
    create_reps("uniprot")
    create_reps("pubmed")
    create_reps("uniprotpubmed")
