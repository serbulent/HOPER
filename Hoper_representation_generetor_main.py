import yaml
import pandas as pd
import os
import glob
import tqdm
from sklearn.utils import shuffle
import os
import pickle
import imghdr
import sys


# upload yaml file
path = os.getcwd()
sys.path.append(path + "/case_study/bin/")
stream = open(path + "/Hoper_representation_generetor.yaml", "r")
data = yaml.safe_load(stream)
module_name=data["parameters"]["module_name"]

isDirected =False
if "PPI" in data["parameters"]["choice_of_module"] :
  
  from ppi_representations import Node2vec
  from ppi_representations import HOPE
  edge_f=data["parameters"]["interaction_data_path"][0]
  protein_id=pd.read_csv(data["parameters"]["protein_id_list"][0])
  node2vec_parameters=data["parameters"]["node2vec_module"]["parameter_selection"]
  os.system("conda env create -f /ppi_representations/ppi_environment.yml")
  if "Node2vec" in data["parameters"]["choice_of_representation_name"]:
    Node2vec.node2vec_repesentation_call(edge_f,protein_id,isDirected,node2vec_parameters["d"],node2vec_parameters["p"],node2vec_parameters["q"])
  if "HOPE" in  data["parameters"]["choice_of_representation_name"]:
    HOPE.hope_repesentation_call(edge_f,protein_id,isDirected,data["parameters"]["HOPE_module"]["parameter_selection"]["d"],data["parameters"]["HOPE_module"]["parameter_selection"]["beta"])

if "text" in data["parameters"]["choice_of_module"] :
  os.system("pip install sent2vec")
  os.system("conda env create -f text_representations/text_representations.yml")
  os.system("conda activate textrep")
  if "generate" in data["parameters"]["choice_of_process"]:
    os.system('python text_representations/representation_generation/createtextrep.py --' + data["parameters"]["generate_module"]["choice_of_representation_type"][0] + ' -upfp ' + data["parameters"]["generate_module"]["uniprot_files_path"][0] + ' -pmfp ' + data["parameters"]["generate_module"]["pubmed_files_path"][0])
  if "visualize" in  data["parameters"]["choice_of_process"]:
    os.system('python text_representations/result_visualization/visualize_results.py -' + data["parameters"]["visualize_module"]["choice_of_visualization_type"][0] + ' -rfp ' + data["parameters"]["visualize_module"]["result_files_path"][0])

if "Preprocessing" in data["parameters"]["choice_of_module"] :
  os.system("conda env create -f /text_representations/preprocess/preprocess.yml")
  os.system('python /text_representations/preprocess/preprocess_main.py')
