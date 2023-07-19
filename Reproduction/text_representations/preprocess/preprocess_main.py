#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm
from Bio import Entrez
from Bio.Entrez import efetch
import pdb

from preprocess import CC_subsection_extractor
from preprocess import subsection
from preprocess import parsing_pubmed_ids
from preprocess import extracting_abstracts



CC_subsection_extractor.main()
CC_subsection_extractor.extract_relevant_info_from_uniprot(record)
subsection.main()
subsection.removing_parentheses()
subsection.removing_dots()
subsection.removing_spaces()
parsing_pubmed_ids.main()
extracting_abstracts.fetch_pubmed_abstracts()