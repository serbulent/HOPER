# Sequence representations (ProtT5-XL)

`prott5xl.py` computes one 1024-dimensional vector per protein with the ProtT5-XL encoder
(mean of the per-residue embeddings). It runs in the `prott5xl` environment created by `create_env.sh`.

```shell
conda activate prott5xl
python sequence_representations/prott5xl.py --input proteins.csv --output outputs/prott5_bfd.csv
```

- **Input**: a CSV with an identifier column (`Entry` or `ID`; or `--id_column`) and a `Sequence` column, or a FASTA file.
- **Output**: a multi-column CSV (`Entry`, `0` ... `1023`), the format used by the other HOPER modules
  (e.g. `--seq_csv` of the multimodal autoencoders).
- **Model** (`--model`):
  - `bfd` (default) - [Rostlab/prot_t5_xl_bfd](https://huggingface.co/Rostlab/prot_t5_xl_bfd). The sequence vectors in the
    example data (`data/hoper_sequence_representations/T5_UNIPROT_HUMAN.csv`) were produced with this model
    (cosine similarity ~0.9999 with vectors computed by this script), so new proteins end up in the same space.
  - `uniref50` - [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50).
- Only the encoder is used. On first run its weights (~4.8 GB) are downloaded from the model's safetensors
  checkpoint and cached in `sequence_representations/models/` (or `--model_dir` / `$HOPER_MODEL_DIR`).
  Loading needs ~6 GB of RAM; the full 11.3 GB checkpoint is never loaded.
- Residues are space-separated and `U`, `Z`, `O`, `B` are mapped to `X`, as ProtT5 expects.
  Sequences longer than `--max_length` (default 2048) are skipped with a warning.
- CPU by default; `--device cuda` (or `HOPER_DEVICE=cuda`) uses a GPU in half precision.
  On CPU a 300-residue protein takes about one second.

Also available from the launcher (`choice_of_module: [sequence]`, see the main README).
