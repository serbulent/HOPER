#!/usr/bin/env bash
# Downloads the HOPER example data (data.zip, ~670 MB) and, with --uniprot, the UniProt/Swiss-Prot
# XML used by the Preprocessing module (~850 MB). Run from anywhere; files go to the repository root.
set -euo pipefail
cd "$(dirname "$0")"

DATA_ID=1R7jRfnBWmO6i6S1vqQd6zZt2-kcK6Eom
UNIPROT_ID=1fOu7cWX9f-B-Ro41VvLGgG8eyGhV8IwD

gdrive() { # <file id> <output>
  curl -fL --retry 3 -o "$2" "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"
}

PY=python3
command -v python3 >/dev/null 2>&1 || PY="conda run -n hoper python"

if [ ! -d data ]; then
  [ -f data.zip ] || gdrive "$DATA_ID" data.zip
  $PY -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"
fi

# Text representation inputs: README step "copy uniprot and pubmed text files to .../representation_generation/data/"
RG=text_representations/representation_generation/data
if [ ! -d "$RG/uniprot" ] || [ ! -d "$RG/pubmed" ]; then
  $PY -c "import zipfile; [zipfile.ZipFile('data/text_representations/%s.zip' % z).extractall('$RG') for z in ('uniprot', 'pubmed')]"
fi
RV=text_representations/result_visualization/result_files
if [ ! -d "$RV/results" ]; then
  $PY -c "import zipfile; zipfile.ZipFile('data/text_representations/results.zip').extractall('$RV')"
fi

if [ "${1:-}" = "--uniprot" ] && [ ! -f uniprot_sprot.xml.gz ]; then
  gdrive "$UNIPROT_ID" uniprot_sprot.xml.gz
fi
echo "Data ready."
