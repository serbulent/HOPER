import pandas as pd
import os
from tqdm import tqdm
import nlu
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import gc


ufiles_path = ''
pfiles_path = ''

path = "/media/DATA/home/muammer/HOPER/"

def convert_dataframe_to_multi_col(representation_dataframe):
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(representation_dataframe['Vector'])
    multi_col_representation_vector = pd.merge(left=entry,right=vector,left_index=True, right_index=True)
    return multi_col_representation_vector

def create_reps(tp):
    files = os.listdir(pfiles_path)
    file_list = []

    # BioBERT model ve tokenizer'ı yükleme
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    data = []
    print('Generating ' + tp + ' embeddings...')
    #for i in tqdm(range(2)):
    for i in tqdm(range(len(files))):
        file_content = ""
        #print(files[i])
        if tp == 'uniprot':
            with open(os.path.join(ufiles_path, files[i])) as contentu:
            #contentu = open(ufiles_path + files[i])
                file_content = contentu.read()
        elif tp == 'pubmed':
            with open(os.path.join(pfiles_path, files[i])) as contentp:
            #contentp = open(pfiles_path + files[i])
                file_content = contentp.read()
        elif tp == 'uniprotpubmed':
            with open(os.path.join(ufiles_path, files[i])) as contentu, open(os.path.join(pfiles_path, files[i])) as contentp:
            #contentu = open(ufiles_path + files[i])
            #contentp = open(pfiles_path + files[i])
                file_content = contentu.read() + contentp.read()
        
        inputs = tokenizer(file_content, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Model ile token embeddinglerini hesapla
        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS] token embeddingini al (genellikle tüm metnin temsili olarak kullanılır)
            cls_embedding = outputs.last_hidden_state[:, 0, :] 
            flat_embedding = cls_embedding.squeeze().numpy().flatten()
            data.append([files[i][:-4], flat_embedding])
            
        del cls_embedding, outputs
        gc.collect()

    df = pd.DataFrame(data, columns=['Entry', 'Vector']) 
    #print(df.head())
    df = convert_dataframe_to_multi_col(df)
    df.to_csv(os.path.join(path,'text_representations/representation_generation/biobert_representations/' + tp + '_biobert_embeddings_multi_col.csv'), index = False)
   
def main():
    create_reps("uniprot")
    #create_reps("pubmed")
    #create_reps("uniprotpubmed")
