# Preprocess
The aim of preprocess is extracting and editing the information of the xml files of the proteins.
Firstly, the UniProt (Swiss-Prot) database is read and the information of the subsections in the General annotation (Comments) is extracted.
Secondly, PubMed references in the text are removed.
Finally, PubMed ids and abstracts of human proteins are parsed and saved.

# How to Run Preprocess
1. Install HOPER and download `uniprot_sprot.xml.gz` (main [README](../../README.md#installation):
   `bash create_env.sh`, `bash download_data.sh --uniprot`).
2. In `Hoper_representation_generetor.yaml` set `choice_of_module: [Preprocessing]` (`uniprot_dir: ./uniprot_sprot.xml.gz`).
3. From the repository root:

```shell
conda activate hoper
python Hoper_representation_generetor_main.py
```

The local steps take ~10-15 minutes for the whole of Swiss-Prot (570,157 entries) and write ~9 GB (about 2.3 million
small files). The last step downloads the PubMed abstracts of ~20,000 human proteins from NCBI, which takes several
hours; it runs only when `HOPER_ENTREZ_EMAIL` is set to your e-mail address (NCBI policy); `NCBI_API_KEY` is used if set.

# Dependencies
`text_representations/preprocess/hoper_preprocess.yml` (environment `hoper_preprocess`, created by `create_env.sh`).

# Output files
Written to `text_representations/preprocess/data/`: `uniprot_subsections/`, `uniprot_par/`, `uniprot_dot/`,
`uniprot_space/` (one text file per protein after each cleaning step), `human_pubmed_ids/` and
`human_pubmed_abstracts/`.
