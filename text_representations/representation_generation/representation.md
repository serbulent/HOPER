# Representation generation

`createtextrep.py` creates text-based protein representations from UniProt and PubMed text files (one `<UniProt id>.txt`
file per protein in each folder): TF-IDF, BioBERT, BioSentVec, BioWordVec and OpenAI embeddings. For each method
three variants are produced: `uniprot`, `pubmed` and `uniprotpubmed` (both texts concatenated).

## Setup

`bash create_env.sh` creates the `HOPER_textrepresentations` environment and `bash download_data.sh` places the
example texts (20,365 human proteins) in `data/uniprot/` and `data/pubmed/` of this folder. See the main
[README](../../README.md#installation).

## Running

From the repository root with the launcher (`choice_of_module: [text]`, `choice_of_process: [generate]`, see the
main README), or directly:

```shell
conda activate HOPER_textrepresentations
python text_representations/representation_generation/createtextrep.py --tfidf \
  -upfp text_representations/representation_generation/data/uniprot/ \
  -pmfp text_representations/representation_generation/data/pubmed/
```

### Options

| Option | |
|---|---|
| `-tfidf`, `--tfidf` | TF-IDF |
| `-biobert`, `--biobert` | BioBERT (`dmis-lab/biobert-base-cased-v1.1`, downloaded from Hugging Face on first use) |
| `-bsv`, `--biosentvec` | BioSentVec (~22 GB model, must fit in RAM) |
| `-bwv`, `--biowordvec` | BioWordVec (~13 GB model, must fit in RAM) |
| `-openai`, `--openai` | OpenAI `text-embedding-3-large` (needs `OPENAI_API_KEY`) |
| `-a`, `--all` | all of the above |
| `-upfp`, `--uniprotfilespath` | folder with the UniProt text files (required) |
| `-pmfp`, `--pubmedfilespath` | folder with the PubMed text files (required) |
| `-mdw y`, `--model_download y` | download the BioSentVec/BioWordVec models to `models/` if they are missing |

The BioSentVec/BioWordVec models can also be downloaded beforehand into `models/`:

```shell
curl -L -o text_representations/representation_generation/models/BioSentVec_PubMed_MIMICIII-bigram_d700.bin https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin
curl -L -o text_representations/representation_generation/models/BioWordVec_PubMed_MIMICIII_d200.bin https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
```

## Output

Written to `<method>_representations/` in this folder; all CSVs are multi-column (`Entry`, `0`, `1`, ...).

- TF-IDF (`tfidf_representations/`): SVD-reduced vectors `<type>_tfidf_vectors_svd{256,512,1024,2048}.csv` and the
  full sparse matrix `<type>_tfidf_vectors.npz` with `<type>_tfidf_entries.csv` (rows) and
  `<type>_tfidf_vocabulary.csv` (columns). `HOPER_TFIDF_DENSE_CSV=1` also writes the full matrix as a dense CSV
  (~8 GB of RAM for the full data set). The full data set takes about 20 minutes on CPU.
- BioBERT: `biobert_representations/<type>_biobert_embeddings_multi_col.csv` (768-d).
- BioSentVec: `biosentvec_representations/<type>_biosentvec_vectors_multi_col.csv` (700-d).
- BioWordVec: `biowordvec_representations/<type>_biowordvec_vectors_multi_col.csv` (200-d).
- OpenAI: `openai_representations/<type>_openai_large_vectors_multi_col.csv` (3072-d).
