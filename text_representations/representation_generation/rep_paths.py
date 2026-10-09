"""Paths and helpers shared by the text representation generators.

All model and output locations are resolved relative to this directory, so the
generators work no matter which directory they are launched from.
"""
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(MODULE_DIR, "models")

BIOSENTVEC_MODEL = os.path.join(MODELS_DIR, "BioSentVec_PubMed_MIMICIII-bigram_d700.bin")
BIOWORDVEC_MODEL = os.path.join(MODELS_DIR, "BioWordVec_PubMed_MIMICIII_d200.bin")
BIOSENTVEC_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
BIOWORDVEC_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin"


def output_dir(name):
    """Return (and create) <module dir>/<name>_representations."""
    path = os.path.join(MODULE_DIR, name + "_representations")
    os.makedirs(path, exist_ok=True)
    return path


def read_text(directory, file_name):
    with open(os.path.join(directory, file_name), encoding="utf-8", errors="replace") as handle:
        return handle.read()


def ensure_nltk_data():
    """Download the NLTK resources used by the tokenizers if they are missing."""
    import nltk
    for resource, path in (("stopwords", "corpora/stopwords"), ("punkt", "tokenizers/punkt"),
                           ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def require_model(path, name):
    """Fail early with a clear message instead of loading a missing or placeholder model file."""
    if not os.path.isfile(path) or os.path.getsize(path) < 1024 * 1024:
        raise SystemExit(
            "{} model not found at {}.\nDownload it (see README) or run with -mdw y.".format(name, path))
