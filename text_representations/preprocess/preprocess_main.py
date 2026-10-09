#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm import tqdm
from Bio import Entrez
from Bio.Entrez import efetch
import pdb

import CC_subsection_extractor
import subsection
import parsing_pubmed_ids
import extracting_abstracts



CC_subsection_extractor.main()
subsection.main()
#subsection.removing_parentheses()
#subsection.removing_dots()
#subsection.removing_spaces()
parsing_pubmed_ids.main()
if extracting_abstracts.entrez_email:
    extracting_abstracts.main()
else:
    print("HOPER_ENTREZ_EMAIL is not set: skipping the PubMed abstract download "
          "(~20k NCBI requests, several hours). Set it to your e-mail address to run this step.")
