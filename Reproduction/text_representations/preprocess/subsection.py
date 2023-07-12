#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm

#removing PubMed references in the text; removing parentheses, dots and spaces in order

yaml_file_path=os.getcwd()                                                  #upload yaml file
stream = open(yaml_file_path+'/2_subsection_edition_config.yaml', 'r')
data = yaml.safe_load(stream)

def removing_parentheses():      #removing parentheses including PubMed

   files_par=os.listdir(data["parameters"]["removing_parentheses"]["files_par"])      #output files including PubMed subsections

   for i in tqdm(files_par):                                           
     old_file_par=open(data["parameters"]["removing_parentheses"]["old_file_par"] + i, "r")
     new_file_par=open(data["parameters"]["removing_parentheses"]["new_file_par"] + i, "w")
     data1 = old_file_par.read()
     data2=(data1.split("\n\n"))
     for j in range(0, len(data2)):
       words_count = len(data2[j].split())      #counting words in every paragraph
       split_words = data2[j].split()           #splitting the words in the paragraphs
       for i in range(0, words_count):
           if "PubMed:" in split_words[i]:
               if ")" in split_words[i]:
                  split_words[i]= "."          
               else:
                   split_words[i]=""

           new_file_par.write(split_words[i]+ " ")
       new_file_par.write("\n\n")
       
   removing_dots()    



def removing_dots():     #removing dots

   files_dot=os.listdir(data["parameters"]["removing_dots"]["files_dot"])       #files saved from the removing_parantheses() function
   
   for j in (files_dot):
      old_file_dot=open(data["parameters"]["removing_dots"]["old_file_dot"] + j, "r")    
      new_file_dot=open(data["parameters"]["removing_dots"]["new_file_dot"] + j, "w")
      for line in old_file_dot:
         new_file_dot.write(line.replace(' .', '.'))
         
   removing_spaces()
   
         
def removing_spaces():     #removing spaces

   files_space=os.listdir(data["parameters"]["removing_spaces"]["files_space"])

   for j in (files_space):
      old_file_space=open(data["parameters"]["removing_spaces"]["old_file_space"]+ j, "r")
      new_file_space=open(data["parameters"]["removing_spaces"]["new_file_space"] + j, "w")
      for line in old_file_space:
          new_file_space.write(line.replace('  ', ''))         


def main():
   removing_parentheses()       
         
if __name__ == "__main__":
    main()