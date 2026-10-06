'''
This script provides a convenient way to create different types of text representations by specifying the desired representation techniques and the paths to the input files via command-line arguments.
Only the generator modules that are selected are imported, so e.g. TF-IDF does not need the BioSentVec/BioWordVec/OpenAI dependencies.
'''

import argparse
import importlib
import os
import urllib.request

import rep_paths

parser = argparse.ArgumentParser(description='Create text representations')
parser.add_argument("-tfidf","--tfidf", action='store_true', help="Create TFIDF representations")
parser.add_argument("-biobert","--biobert", action='store_true', help="Create bioBERT representations")
parser.add_argument("-bsv", "--biosentvec", action='store_true',  help="Create biosentvec representations")
parser.add_argument("-bwv", "--biowordvec", action='store_true',  help="Create biowordvec representations")
parser.add_argument("-openai","--openai", action='store_true', help="Create OpenAI representations (needs OPENAI_API_KEY)")
parser.add_argument("-upfp", "--uniprotfilespath", required=True,  help="Path for the uniprot files")
parser.add_argument("-pmfp", "--pubmedfilespath", required=True,  help="Path for the pubmed files")
parser.add_argument("-mdw", "--model_download", default="n", help="y: download biosentvec and biowordvec pre-trained models automatically")
parser.add_argument("-a", "--all", action='store_true',  help="Create all representations")
args = parser.parse_args()
print(args)


def download_model(url, path):
    """Download a pre-trained model to its exact file name (skipped when it already exists)."""
    if os.path.isfile(path) and os.path.getsize(path) > 1024 * 1024:
        print("Model already present: " + path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading " + url + " -> " + path)
    tmp_path = path + ".part"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)


def run(module_name, label):
    print("\n\n Creating " + label + " representations...\n")
    module = importlib.import_module(module_name)
    module.ufiles_path = args.uniprotfilespath
    module.pfiles_path = args.pubmedfilespath
    module.main()


if args.tfidf or args.all:
    run("create_tfidf", "tfidf")

if args.biobert or args.all:
    run("create_biobert", "biobert")

if args.openai or args.all:
    run("create_openai", "OpenAI")

if args.biosentvec or args.all:
    if args.model_download == "y":
        download_model(rep_paths.BIOSENTVEC_URL, rep_paths.BIOSENTVEC_MODEL)
    run("create_biosentvec", "biosentvec")

if args.biowordvec or args.all:
    if args.model_download == "y":
        download_model(rep_paths.BIOWORDVEC_URL, rep_paths.BIOWORDVEC_MODEL)
    run("create_biowordvec", "biowordvec")
