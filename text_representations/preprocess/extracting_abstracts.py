#!/usr/bin/env python

import yaml
from Bio import Entrez
from Bio.Entrez import efetch
import os
import pdb
from tqdm import tqdm

# NCBI requires a real contact e-mail; an optional API key raises the rate limit from 3 to 10 requests/s.
entrez_email = os.environ.get("HOPER_ENTREZ_EMAIL")
if os.environ.get("NCBI_API_KEY"):
    Entrez.api_key = os.environ["NCBI_API_KEY"]

yaml_file_path=os.getcwd()                                                  #upload yaml file
stream = open(os.environ.get("HOPER_CONFIG", os.path.join(yaml_file_path, "Hoper_representation_generetor.yaml")), "r")

data = yaml.safe_load(stream)

#extracting PubMed abstracts only for Human

def fetch_pubmed_abstracts(id_list):
        Entrez.email = entrez_email  
        handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=id_list)
        results = Entrez.read(handle)
        return results
def main():
    if not entrez_email:
        raise SystemExit("Set HOPER_ENTREZ_EMAIL to your e-mail address to download PubMed abstracts (NCBI usage policy).")

    files=os.listdir(os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_ids"))   #files saved from the parsing_pubmed_ids.py file
   
    error=open(os.path.join(yaml_file_path, "text_representations/preprocess/data/pubmed_new_abstracts","error_abstract_human_2.txt"), "w")

    for protein_id in tqdm(files):
        try:
            with open(os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_ids" , protein_id), "r") as protein_pubmed_id_file:
                pubmed_ids = protein_pubmed_id_file.readlines()
                papers = fetch_pubmed_abstracts(pubmed_ids)
        
            os.makedirs(os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_abstracts"), exist_ok=True)
            with open(os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_abstracts" ,protein_id), "w") as protein_abstract_file:
               for pubmed_paper_index in range (0, len(pubmed_ids)):
                   if 'Abstract' in papers['PubmedArticle'][pubmed_paper_index]['MedlineCitation']['Article']:
                 #pdb.set_trace()
                     pubmed_abstract_text = papers['PubmedArticle'][pubmed_paper_index]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                     protein_abstract_file.write(str(pubmed_abstract_text)+ "\n\n")
             
             
        except Exception as e:
            print(e)