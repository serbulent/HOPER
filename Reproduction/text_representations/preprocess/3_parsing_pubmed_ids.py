#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm
import pdb


error=open("/media/DATA/amine/pubmed_yeni_abstractlar/error_abstract_human_2.txt", "w")

yaml_file_path=os.getcwd()                                                  #upload yaml file
stream = open(yaml_file_path+'/2_subsection_edition_config.yaml', 'r')
data = yaml.safe_load(stream)

def extract_relevant_info_from_uniprot(uniprot_record):
    # print(record)
    try:
       
         if os.path.isfile(["parameters"]["files"] + record.id + '.txt')==False:
            
            newfile = open(["parameters"]["files"] + record.id + '.txt','w')
            #print(record.id)
            len_of_ids=len(uniprot_record.annotations["references"])
            
            for i in range(0,len_of_ids):
              
               ref = uniprot_record.annotations['references'][i]
               if (ref.pubmed_id)!="":
                           
                 newfile.write(str(ref.pubmed_id) + "\n")
                   
    except Exception as e:
        error.write(record.id)
        error.write("\n")
        error.write(str(e))
        error.write("\n\n")
        # print(record.id)

with gzip.open('/media/DATA/amine/uniprot_sprot.xml.gz', 'rb') as handle:
    # file=handle.read()
    # for record in SeqIO.parse(file, "uniprot-xml"):
    for record in tqdm(SeqIO.UniprotIO.UniprotIterator(handle)): 
       if (record.annotations["organism"]=="Homo sapiens (Human)"):
         extract_relevant_info_from_uniprot(record)