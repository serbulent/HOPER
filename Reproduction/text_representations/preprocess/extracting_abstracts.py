#!/usr/bin/env python

import yaml
from Bio import Entrez
from Bio.Entrez import efetch
import os
import pdb
from tqdm import tqdm

entrez_email = "aaaaaa@ogr.ktu.edu.tr" 

yaml_file_path=os.getcwd()                                                  #upload yaml file
stream = open(yaml_file_path+'/4_extracting_abstracts_config.yaml', 'r')
data = yaml.safe_load(stream)

#extracting PubMed abstracts only for Human

def fetch_pubmed_abstracts(id_list):
        Entrez.email = entrez_email  
        handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=id_list)
        results = Entrez.read(handle)
        return results

files=os.listdir(data["parameters"]["files_ids"])   #files saved from the parsing_pubmed_ids.py file
error=open(data["parameters"]["error_file"], "w")

for protein_id in tqdm(files):
    try:
        with open(data["parameters"]["files_ids"] + protein_id, "r") as protein_pubmed_id_file:
            pubmed_ids = protein_pubmed_id_file.readlines()
            papers = fetch_pubmed_abstracts(pubmed_ids)
            
        with open(data["parameters"]["files_abstracts"] + protein_id, "w") as protein_abstract_file:
           for pubmed_paper_index in range (0, len(pubmed_ids)):
               if 'Abstract' in papers['PubmedArticle'][pubmed_paper_index]['MedlineCitation']['Article']:
                 #pdb.set_trace()
                 pubmed_abstract_text = papers['PubmedArticle'][pubmed_paper_index]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                 protein_abstract_file.write(str(pubmed_abstract_text)+ "\n\n")
             
             
    except Exception as e:
        print(e)