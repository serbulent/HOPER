import json
import os
import sys

import pandas as pd

# Allow ``python utils/fuse_representation.py`` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RepresentationFusion


def make_fuse_representation(representation_files, min_fold_num, representation_names):
  representation_file_list=[]
  for rep_file in representation_files:
    directory, rep_file_name = os.path.split(rep_file)
    print("loading " + rep_file_name + "...")
    representation_file_list.append(
                    pd.read_csv(rep_file)
                )
  if min_fold_num is not None and 0 < min_fold_num <= len(representation_file_list):
    min_fold_number=min_fold_num
  else:
    min_fold_number = len(
                    representation_file_list
                )

  representation_dataframe = (
                RepresentationFusion.produce_fused_representations(
                    representation_file_list,
                    min_fold_number,
                    representation_names,
                )
            )
  os.makedirs("./data", exist_ok=True)
  fuse_representation_path=os.path.join("data","_".join(
                [str(representation) for representation in representation_names])+"_binary_fused_representations_dataframe_multi_col.csv")
  pd.DataFrame(representation_dataframe).to_csv(
                fuse_representation_path,
                index=False,
            )
  print("saved " + fuse_representation_path)
  return representation_dataframe


def _parse_list(value):
  """Accept a JSON list (as passed by the HOPER launcher) or a comma separated string."""
  try:
    parsed = json.loads(value)
    if isinstance(parsed, list):
      return parsed
  except ValueError:
    pass
  return [item.strip() for item in value.strip("[]").split(",") if item.strip()]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python utils/fuse_representation.py '<json list of files>' <min_fold_number|None> '<json list of names>'")
        sys.exit(1)
    files = _parse_list(sys.argv[1])
    min_fold = None if sys.argv[2] in ("None", "none", "") else int(sys.argv[2])
    names = _parse_list(sys.argv[3])
    make_fuse_representation(files, min_fold, names)
