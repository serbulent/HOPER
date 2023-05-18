import pandas as pd
import os
path = os.path.dirname(os.getcwd())
from Preprocess import RepresentationFusion
class ModelParameterClass:
  def __init__(self,choice_of_task_name,fuse_representations,prepare_datasets,model_training_test,prediction):
    self.choice_of_task_name=choice_of_task_name
    self.fuse_representations=fuse_representations
    self.prepare_datasets=prepare_datasets
    self.model_training_test=model_training_test
    self.prediction=prediction
    
  def make_fuse_representation(self):
    representation_file_list=[]
    path = os.path.dirname(os.getcwd())
    for rep_file in self.fuse_representations["representation_files"]:
      rep_file_name = rep_file.split("/")[-1]
      print("loading " + rep_file_name + "...")
      representation_file_list.append(
                    pd.read_csv(rep_file)
                )
    if self.fuse_representations["min_fold_number"]!="None" and self.fuse_representations["min_fold_number"]>0:
      self.min_fold_number=self.fuse_representations["min_fold_number"]
    else:
      self.min_fold_number = len(
                    representation_file_list
                )
  
    #import pdb
    #pdb.set_trace()
    representation_dataframe = (
                RepresentationFusion.produce_fused_representations(
                    representation_file_list,
                    self.min_fold_number,
                    self.fuse_representations[
            "representation_names"
        ],
                )
            )
    pd.DataFrame(representation_dataframe).to_csv(
                path
                + "/results/"
                + "_".join(
                [str(representation) for representation in self.fuse_representations["representation_names"]])
                + "_binary_fused_representations_dataframe_multi_col.csv",
                index=False,
            )
    return representation_dataframe
    
  