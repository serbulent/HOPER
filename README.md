# HOPER (Holistic Protein Representation)

- Holistic protein representation uses  multimodal learning to predict protein functions with low amount of training data. 

- Representation vectors are created using protein sequence, protein text and protein-protein interaction data types to achieve this goal.

- The rationale behind incorporating protein-protein interactions into our holistic protein representation model is the assumption 
that interacting proteins are likely to act in the same biological process. These proteins are also likely to be located in the same cellular compartment. 

- Text-based protein representations calculated with pre-trained natural language processing models.

<img width="729" height="960" alt="HOPER_Manuscript_Figure1" src="https://github.com/user-attachments/assets/d3d2e2fe-2579-45da-a151-f1c0c8f9327a" />

The overview of HOPER. We first generated protein representations (embeddings) independently using three different modalities (i.e., protein sequence, protein-protein interaction, and protein-related text). Then, we benchmarked them to find the best-performing representation model for each modality in the context of protein function prediction in a low-data setting. After that, we constructed multimodal learning models that take representations of independent modalities as input and produce a holistic embedding by leveraging their relationships. Finally, as a use-case study, we predicted new tumor immune-escape proteins in the lung adenocarcinoma using our model and discussed findings.

# Installation

## Requirements

- Linux x86-64 (on Windows use WSL2), `git`, and [Miniforge](https://github.com/conda-forge/miniforge) / Miniconda / Anaconda.
  No system C/C++ compiler or `sudo` is needed.
- Disk: ~35 GB for the conda environments, ~1 GB for the example data
  (+ ~0.85 GB for `uniprot_sprot.xml.gz` and ~9 GB of preprocessing output if you run *Preprocessing*).
- RAM: 8 GB is enough for every step in this README (ProtT5-XL needs ~6 GB) except BioSentVec (~22 GB model) and
  BioWordVec (~13 GB model), which must fit in memory.
- A GPU is optional. All modules run on CPU by default; set `HOPER_DEVICE=cuda` to use a GPU.

## Steps

```shell
git clone https://github.com/serbulent/HOPER.git
cd HOPER
bash create_env.sh          # creates/updates all environments, installs GEM, builds node2vec (prints a summary)
bash download_data.sh       # example data (~670 MB); add --uniprot for uniprot_sprot.xml.gz (needed by Preprocessing)
bash tests/smoke_test.sh    # optional: runs every step below on small inputs and checks the outputs
```

`create_env.sh` creates these environments (each module runs in its own one):

| Environment | Used by |
|---|---|
| `hoper` | `Hoper_representation_generetor_main.py` (launcher, only needs `pyyaml`) |
| `hoper_PPI` | Node2vec, HOPE ([GEM](https://github.com/palash1992/GEM) @ `213189b`, SNAP node2vec) |
| `hoper_preprocess` | UniProt / PubMed preprocessing |
| `HOPER_textrepresentations` | TF-IDF, BioBERT, BioSentVec, BioWordVec, OpenAI representations and result visualization |
| `hoper_case_study_env` | `fuse_representations`, `case_study_main.py` |
| `HoloProtRep-AE` | SimpleAE, MultiModalAE, TransferAE |
| `prott5xl` | ProtT5-XL sequence representations |
| `hoper_build` | only used to compile SNAP node2vec into `ppi_representations/bin/` |

`download_data.sh` places the data where the modules expect it: `./data/`, the UniProt and PubMed text files in
`text_representations/representation_generation/data/{uniprot,pubmed}/` and the benchmark results in
`text_representations/result_visualization/result_files/results/`.

# How to run HOPER

Representation modules are run by the launcher with the configuration file `Hoper_representation_generetor.yaml`
(edit `choice_of_module` and the matching section; a different config file can be passed as an argument):

```shell
conda activate hoper
python Hoper_representation_generetor_main.py            # or: python Hoper_representation_generetor_main.py my_config.yaml
```

The launcher runs each selected module in its environment and stops with an error message if a step fails.

### PPI representations (Node2vec, HOPE)

More information: [ppi_representations/readme.md](ppi_representations/readme.md)

```yaml
parameters:
    choice_of_module: [PPI]
    choice_of_representation_name:  [Node2vec,HOPE]
    interaction_data_path:  [./data/hoper_PPI/PPI_example_data/example.edgelist]
    protein_id_list:  [./data/hoper_PPI/PPI_example_data/proteins_id.csv]
    is_directed: false
    node2vec_module:
        parameter_selection:
            d:  [10]  
            p:  [0.25]
            q:  [0.25]
    HOPE_module:
        parameter_selection:
            d:  [5]
            beta:  [0.00390625]
```

Output: `data/Node2vec_d_<d>_p_<p>_q_<q>.pkl` and `data/HOPE_d_<d>_beta_<beta>.pkl` (columns `Entry`, `Vector`).

### Sequence representations (ProtT5-XL)

More information: [sequence_representations/readme.md](sequence_representations/readme.md)

```yaml
parameters:
    choice_of_module: [sequence]
    sequence_module:
        input_path: ./sequence_representations/example_sequences.fasta   # FASTA, or CSV with Entry + Sequence
        output_path: ./outputs/prott5_bfd_representation.csv
        model: bfd          # bfd (matches the example data) or uniref50
        batch_size: 8
```

or directly:

```shell
conda activate prott5xl
python sequence_representations/prott5xl.py --input proteins.fasta --output outputs/prott5_bfd_representation.csv
```

Output: a multi-column CSV (`Entry`, 1024 columns). The default model, `Rostlab/prot_t5_xl_bfd`, is the one used
for the sequence vectors in the example data (`data/hoper_sequence_representations/T5_UNIPROT_HUMAN.csv`).
The encoder weights (~4.8 GB) are downloaded on first use to `sequence_representations/models/`.

### Text preprocessing (UniProt / PubMed)

More information: [text_representations/preprocess](text_representations/preprocess). Needs `bash download_data.sh --uniprot`.

```yaml
parameters:
    choice_of_module: [Preprocessing]
    uniprot_dir: ./uniprot_sprot.xml.gz 
```

Output goes to `text_representations/preprocess/data/`. Parsing all of Swiss-Prot takes ~10-15 minutes and ~9 GB of disk
(about 2.3 million small files; on WSL2 with 8 GB RAM the VM can become unresponsive for a few minutes while they are written).
The last step downloads PubMed abstracts for ~20,000 human proteins from NCBI (several hours); it runs only when
`HOPER_ENTREZ_EMAIL` is set to your e-mail address (NCBI policy). `NCBI_API_KEY` is used if set.

### Text representations

More information: [text_representations/representation_generation/README.md](text_representations/representation_generation/README.md)

```yaml
parameters:
    choice_of_module: [text]
    choice_of_process:  [generate,visualize]
    generate_module:
        choice_of_representation_type:  [tfidf]     # tfidf, biobert, biosentvec, biowordvec, openai or all
        uniprot_files_path:  [./text_representations/representation_generation/data/uniprot/]
        pubmed_files_path:  [./text_representations/representation_generation/data/pubmed/]
        model_download: y
    visualize_module:
        choice_of_visualization_type:  [a]
        result_files_path:  [./text_representations/result_visualization/result_files/results/]
```

Outputs are written to `text_representations/representation_generation/<type>_representations/`.

- `tfidf`: SVD-reduced vectors (`*_tfidf_vectors_svd{256,512,1024,2048}.csv`) and the full sparse matrix
  (`*_tfidf_vectors.npz` + `*_tfidf_entries.csv` + `*_tfidf_vocabulary.csv`). Set `HOPER_TFIDF_DENSE_CSV=1` to also
  write the full matrix as a dense CSV (needs ~8 GB RAM for the full data set).
- `biobert` downloads `dmis-lab/biobert-base-cased-v1.1` from Hugging Face on first use.
- `biosentvec` / `biowordvec`: with `model_download: y` the models are downloaded to
  `text_representations/representation_generation/models/`. Alternatively download them beforehand:

  ```shell
  cd text_representations/representation_generation/models
  curl -L -o BioSentVec_PubMed_MIMICIII-bigram_d700.bin https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin
  curl -L -o BioWordVec_PubMed_MIMICIII_d200.bin https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
  ```
- `openai` needs `OPENAI_API_KEY` in the environment (model `text-embedding-3-large`).
- `visualize` writes figures to `text_representations/result_visualization/figures/` and tables to `.../tables/`.

### Fusing representations

```yaml
parameters:
    choice_of_module: [fuse_representations]
    representation_files: [./data/hoper_case_study_example_data/representation_files/node2vec_d_50_p_0.5_q_0.25_multi_col.csv,./data/hoper_case_study_example_data/representation_files/multi_modal_rep_ae_multi_col_256.csv]
    min_fold_number:  2
    representation_names:  [node2vec,modal_rep_ae]   # same order as representation_files
```

Output: `data/node2vec_modal_rep_ae_binary_fused_representations_dataframe_multi_col.csv`.

### SimpleAE

```yaml
parameters:
    choice_of_module: [SimpleAe]
    representation_path: ./data/hoper_sequence_representations/modal_rep_ae_node2vec_binary_fused_representations_dataframe_multi_col.csv
    simple_ae_module:
        output_dir: ./outputs
        epochs: 400
```

or directly (train / inference):

```shell
conda activate HoloProtRep-AE
python multimodal_representations/simple_ae.py train \
  --fused_rep_path data/hoper_sequence_representations/modal_rep_ae_node2vec_binary_fused_representations_dataframe_multi_col.csv \
  --model_save_path outputs/simple_ae_weights.pth \
  --scaler_save_path outputs/simple_ae_scaler.pkl \
  --output_csv outputs/simple_ae_representation.csv \
  --epochs 400 --batch_size 128 --learning_rate 0.001 --validation_split 0.2 --seed 42 \
  --loss_plot_path outputs/simple_ae_loss.png

python multimodal_representations/simple_ae.py inference \
  --fused_rep_path <multi-column csv with an Entry column> \
  --model_load_path outputs/simple_ae_weights.pth \
  --scaler_load_path outputs/simple_ae_scaler.pkl \
  --output_csv outputs/simple_ae_inference.csv
```

Training on the 14,941 example proteins for 400 epochs takes ~30 minutes on CPU.

### MultiModalAE (sequence + PPI + text)

Inputs are multi-column CSVs with an `Entry` column. The model expects 1024-d sequence (ProtT5, e.g. the output of
`sequence_representations/prott5xl.py`), 500-d PPI and 3072-d text (OpenAI `text-embedding-3-large`) representations.

```shell
conda activate HoloProtRep-AE
python multimodal_representations/multi_odal_representations.py \
  --seq_csv data/sequence_representation.csv \
  --ppi_csv data/ppi_representation.csv \
  --text_csv data/text_representation.csv \
  --representation_dim 512 --epochs 400 --batch_size 128 --lr 0.001 \
  --save_model_path outputs/multimodal_ae_weights.pth \
  --save_csv_path outputs/multimodal_representation.csv \
  --loss_plot_path outputs/multimodal_ae_loss.png
```

### TransferAE (sequence → sequence + text)

TransferAE is initialised from a sequence + text autoencoder, so first train that model, then the transfer model.
In `test` mode only sequence representations are needed.

```shell
conda activate HoloProtRep-AE
python multimodal_representations/multimodal_text_seq.py \
  --seq_csv data/sequence_representation.csv --text_csv data/text_representation.csv \
  --epochs 100 --save_model_path outputs/dual_modal_weights.pth \
  --save_csv_path outputs/fused_dual.csv --loss_plot_path outputs/dual_loss.png

python multimodal_representations/transfer_text_seq.py --mode train \
  --seq_csv data/sequence_representation.csv --text_csv data/text_representation.csv \
  --model_weights outputs/dual_modal_weights.pth \
  --save_model_path outputs/transfer_ae_weights.pth \
  --save_csv_path outputs/transfer_ae_representation.csv \
  --representation_dim 512 --epochs 200 --batch_size 128 --seed 42 \
  --loss_plot_path outputs/transfer_ae_loss.png

python multimodal_representations/transfer_text_seq.py --mode test \
  --seq_csv data/new_sequence_representation.csv \
  --model_weights outputs/transfer_ae_weights.pth \
  --save_csv_path outputs/transfer_ae_test.csv
  # optional: --shift_factors <file> --scaling_factors <file>  (per-dimension (x + shift) * scale normalisation)
```

## Reproducible run of the paper (case study)

Immune-escape prediction; more information: [case_study/readme.md](case_study/readme.md).
`case_study.yaml` is configured for the example data: it prepares the dataset, trains and tests the classifier and
predicts the 1,085 proteins in `rep_dif_ae.csv` (~1 minute on CPU).

```shell
conda activate hoper_case_study_env
python case_study_main.py
```

Outputs: `case_study/case_study_results/{training,test,prediction}/`.

```yaml
parameters:
    choice_of_module: [case_study] 
    module_name: case_study
    choice_of_task_name:  [prepare_datasets,model_training_test,prediction] # also: fuse_representations
    fuse_representations:
        representation_files: [./data/hoper_case_study_example_data/representation_files/node2vec_d_50_p_0.5_q_0.25_multi_col.csv,./data/hoper_case_study_example_data/representation_files/multi_modal_rep_ae_multi_col_256.csv]
        min_fold_number:  2
        representation_names:  [node2vec,modal_rep_ae]
    prepare_datasets:  
        positive_sample_data:  ["./data/hoper_case_study_example_data/prepare_datasets/positive.csv"]
        negative_sample_data:  ["./data/hoper_case_study_example_data/prepare_datasets/neg_data.csv"]
        prepared_representation_file:  ["./data/hoper_case_study_example_data/representation_files/multi_modal_rep_ae_multi_col_256.csv"] 
        representation_names:  [modal_rep_ae]     
    model_training_test:
        representation_names:  [modal_rep_ae]
        scoring_function:  ["f_max"]
        prepared_path:  ["./case_study/case_study_results/modal_rep_ae_binary_data.pickle"]
        classifier_name:  ["Fully_Connected_Neural_Network"]    
    prediction:
        representation_names:  [modal_rep_ae]
        prepared_path:  ["./data/hoper_case_study_example_data/prediction_example_data/rep_dif_ae.csv"]
        classifier_name:  ['Fully_Connected_Neural_Network']         
        model_directory:  ["./case_study/case_study_results/training/modal_rep_ae_Fully_Connected_Neural_Network_binary_classifier.pt"] 
```

The prediction input must have the same representation (and dimension) as the one the model was trained on:
`multi_modal_rep_ae_multi_col_256.csv` and `rep_dif_ae.csv` are both 384-dimensional.
Model files are named `<representation_names>_<classifier>_binary_classifier.pt`.

## License
Copyright (C) 2026

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
