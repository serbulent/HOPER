#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm

#extracting the information of the subsections in the General annotation (Comments) part of the xml files of the proteins
#subsections are function, cofactor, subunit, tissue specificity, induction, domain, PTM, disease
#Main function takes an input file handle and returns an iterator giving SeqRecord objects

yaml_file_path=os.getcwd()                                                  #upload yaml file
stream = open(yaml_file_path+'/1_CC_subsection_extractor_config.yaml', 'r')
data = yaml.safe_load(stream)
uniprot_dir=data["parameters"]["uniprot_dir"]                              #downloading Uniprot database
new_file_dir=data["parameters"]["new_file_dir"]                               #creating directory to save output files         
error_file = open( data["parameters"]["error_file_dir"] + '.txt','w')      #printing the exceptions

def extract_relevant_info_from_uniprot(record):
    
    try:                         
            new_file = open( new_file_dir + record.id + '.txt','w')     #getting the uniprot id of each protein with the record.id and using it in the file name

            if ("comment_function" in record.annotations.keys()):
                functions = record.annotations['comment_function']      #assigning the information of function subsection to the function variable
                for i in range(0, len(functions)):                      #some subsections have more than one input, so we print their information to the related file in order
                    new_file.write(functions[i] + "\n\n")                

            if ("comment_cofactor" in record.annotations.keys()):
                cofactors = record.annotations['comment_cofactor']
                for i in range(0, len(cofactors)):
                    new_file.write(cofactors[i] + "\n\n")

            if ("comment_subunit" in record.annotations.keys()):
                subunits = record.annotations['comment_subunit']
                for i in range(0, len(subunits)):
                    new_file.write(subunits[i] + "\n\n")

            if ("comment_tissuespecificity" in record.annotations.keys()):
                tissuespecificities = record.annotations['comment_tissuespecificity']
                for i in range(0, len(tissuespecificities)):
                    new_file.write(tissuespecificities[i] + "\n\n")

            if ("comment_induction" in record.annotations.keys()):
                inductions = record.annotations['comment_induction']
                for i in range(0, len(inductions)):
                    new_file.write(inductions[i] + "\n\n")

            if ("comment_domain" in record.annotations.keys()):
                domains = record.annotations['comment_domain']
                for i in range(0, len(domains)):
                    new_file.write(domains[i] + "\n\n")

            if ("comment_PTM" in record.annotations.keys()):
                PTMs = record.annotations['comment_PTM']
                for i in range(0, len(PTMs)):
                    new_file.write(PTMs[i] + "\n\n")

            if ("comment_disease" in record.annotations.keys()):
                diseases = record.annotations['comment_disease']
                for i in range(0, len(diseases)):
                    new_file.write(diseases[0])
                    break               
                              
    except Exception as e:
        error_file.write(record.id)
        error_file.write("\n")
        error_file.write(str(e))
        error_file.write("\n\n")
        
def main():
  with gzip.open(uniprot_dir, 'rb') as handle:    
    for uniprot_record in tqdm(SeqIO.UniprotIO.UniprotIterator(handle)):
        extract_relevant_info_from_uniprot(uniprot_record)


if __name__ == "__main__":
    main()