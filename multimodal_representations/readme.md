
# Multimodal Protein Representation Learning

This repository provides a collection of deep-learning-based methods for the integration, compression, and transfer of heterogeneous protein representations.

The framework is designed to combine complementary sources of protein information, including:

- **protein sequence representations,**
- **text-based representations,**
- **protein–protein interaction (PPI) representations.**

Autoencoder-based architectures are used to project these heterogeneous feature spaces into compact latent representations. In addition, transfer-learning models enable representations learned from multimodal data to be transferred to settings in which only sequence information is available.


---

## Repository Structure

| Script | Method | Input Modalities | Main Purpose |
|---|---|---|---|
| `multi_odal_representations.py` | Multimodal Autoencoder | Sequence + Text + PPI | Learning a joint multimodal latent representation |
| `multimodal_text_seq.py` | Dual-Modal Autoencoder | Sequence + Text | Learning a joint sequence–text representation |
| `simple_ae.py` | Autoencoder | Previously fused representation | Further dimensionality reduction of fused embeddings |
| `transfer_ae.py` | Transfer Autoencoder | Sequence + pretrained multimodal model | Transferring information learned from Sequence + Text + PPI |
| `transfer_text_seq.py` | Transfer Autoencoder | Sequence + pretrained dual-modal model | Transferring information learned from Sequence + Text |
| `rep_io.py` | (helper) | | Reading and aligning the representation CSVs |

## Running from the HOPER launcher

All three autoencoders can be run with `Hoper_representation_generetor_main.py` (from the repository root, in the
`hoper` environment) by adding `SimpleAe`, `MultiModalAe` and/or `TransferAe` to `choice_of_module` in
`Hoper_representation_generetor.yaml`. With the example data:

```yaml
    choice_of_module: [text, sequence, MultiModalAe, TransferAe]   # text (tfidf) and sequence produce inputs
    multimodal_ae_module:
        seq_csv: ./data/hoper_sequence_representations/T5_UNIPROT_HUMAN.csv          # ProtT5, 1024-d
        ppi_csv: ./data/hoper_case_study_example_data/representation_files/node2vec_d_50_p_0.5_q_0.25_multi_col.csv  # Node2vec, 50-d
        text_csv: ./text_representations/representation_generation/tfidf_representations/uniprotpubmed_tfidf_vectors_svd1024.csv
        output_dir: ./outputs
        epochs: 100
    transfer_ae_module:                       # TransferAE initialised from the MultiModalAE above
        seq_csv: ./data/hoper_sequence_representations/T5_UNIPROT_HUMAN.csv
        ppi_csv: ./data/hoper_case_study_example_data/representation_files/node2vec_d_50_p_0.5_q_0.25_multi_col.csv
        text_csv: ./text_representations/representation_generation/tfidf_representations/uniprotpubmed_tfidf_vectors_svd1024.csv
        multimodal_weights: ./outputs/multimodal_ae_weights.pth
        test_seq_csv: ./outputs/prott5_bfd_representation.csv   # optional; output of the sequence module
        output_dir: ./outputs
        epochs: 200
```

`TransferAe` trains the sequence-only model from the MultiModalAE weights and, if `test_seq_csv` is set, produces
representations for proteins that only have a sequence representation. `TransferAeSeqText`
(`transfer_ae_seq_text_module`: `seq_csv`, `text_csv`, `test_seq_csv`, `dual_epochs`, `transfer_epochs`) does the same
from a sequence + text autoencoder that it trains first.
Outputs (in `output_dir`): `multimodal_ae_representation.csv`, `transfer_ae_representation.csv`,
`transfer_ae_test_representation.csv` (and `dual_ae_representation.csv`, `transfer_ae_seq_text_*.csv` for the
sequence + text variant), the model weights (`.pth`) and loss curves.

---

# Methodological Overview

## Multimodal Representation Learning

The multimodal architectures employ modality-specific encoders to independently transform sequence, textual, and PPI representations.

The encoded features are subsequently concatenated and projected into a shared latent space through a fusion layer.

---

## Transfer Representation Learning

The transfer-learning models use parameters obtained from previously trained multimodal autoencoders.

The objective is to retain information learned from multiple biological modalities while allowing protein representations to be generated from sequence information alone.

A corresponding sequence–text transfer model is also provided for the dual-modal setting.

---

# Requirements

The scripts run in the `HoloProtRep-AE` environment (`simple_ae_env.yml`), created by `bash create_env.sh`:

```bash
conda activate HoloProtRep-AE
```

The scripts run on CPU by default; set `HOPER_DEVICE=cuda` to use a CUDA GPU.

---

# Input Data Format

Input representations are expected to be provided as CSV files.

Each representation file must contain a protein identifier column named:

```text
Entry
```

followed by numerical representation features.

A general input structure is:

```text
Entry,0,1,2,3,...,n
P12345,0.124,0.532,-0.214,...,0.381
Q67890,0.423,-0.217,0.921,...,-0.114
```

The `Entry` field is used to match proteins across the different representation modalities: only proteins present
in every input file are used. The input dimensions of the multimodal, dual-modal and transfer autoencoders are read
from the files (and, for the transfer model, from the dual-modal weights), so any representation sizes can be
combined. In test/inference mode the inputs must have the same dimensions as during training.

---

# Usage

## 1. Multimodal Autoencoder

### `multi_odal_representations.py`

This script learns a joint protein representation by integrating:

- sequence representations,
- PPI representations,
- text-derived representations.

Each modality is first processed by an independent encoder. The encoded vectors are concatenated and projected into a common latent representation.

### Training

```bash
python multi_odal_representations.py \
    --seq_csv data/sequence.csv \
    --ppi_csv data/ppi.csv \
    --text_csv data/text.csv \
    --representation_dim 512 \
    --batch_size 128 \
    --epochs 100 \
    --validation_split 0.2 \
    --lr 0.001 \
    --save_model_path models/multi_modal_weights.pth \
    --save_csv_path outputs/fused_representation.csv \
    --loss_plot_path outputs/multimodal_loss_curve.png \
    --seed 42
```

### Inference

To generate representations using a previously trained model:

```bash
python multi_odal_representations.py \
    --seq_csv data/sequence.csv \
    --ppi_csv data/ppi.csv \
    --text_csv data/text.csv \
    --representation_dim 512 \
    --load_model_path models/multi_modal_weights.pth \
    --save_csv_path outputs/fused_representation_inference.csv \
    --inference \
    --seed 42
```

### Main Output

```text
fused_representation.csv
```

contains the learned latent multimodal protein representations.

The trained neural-network parameters are stored in:

```text
multi_modal_weights.pth
```

---

## 2. Sequence–Text Multimodal Autoencoder

### `multimodal_text_seq.py`

This script implements a dual-modal autoencoder integrating:

- protein sequence representations,
- text-derived protein representations.

The two feature spaces are independently encoded and subsequently concatenated before projection into a shared latent space.

### Training

```bash
python multimodal_text_seq.py \
    --seq_csv data/sequence.csv \
    --text_csv data/text.csv \
    --representation_dim 512 \
    --batch_size 128 \
    --epochs 100 \
    --validation_split 0.2 \
    --lr 0.001 \
    --save_model_path models/dual_modal_weights.pth \
    --save_csv_path outputs/fused_dual.csv \
    --loss_plot_path outputs/dual_modal_loss_curve.png \
    --seed 42
```

### Inference

```bash
python multimodal_text_seq.py \
    --seq_csv data/sequence.csv \
    --text_csv data/text.csv \
    --representation_dim 512 \
    --load_model_path models/dual_modal_weights.pth \
    --save_csv_path outputs/fused_dual_inference.csv \
    --inference \
    --seed 42
```

### Main Output

```text
fused_dual.csv
```

contains the latent representation jointly learned from sequence and text information.

---

## 3. Autoencoder-Based Dimensionality Reduction

### `simple_ae.py`

This script applies an additional autoencoder to a previously generated fused representation.

Unlike the multimodal models, this architecture receives an already integrated representation as its input and compresses it through a bottleneck layer.

The latent bottleneck representation is used as the final protein embedding.

### Training

```bash
python simple_ae.py train \
    --fused_rep_path data/fused_representation.csv \
    --model_save_path models/autoencoder_best.pth \
    --scaler_save_path models/scaler.pkl \
    --output_csv outputs/simple_ae.csv \
    --epochs 400 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --validation_split 0.2 \
    --seed 42 \
    --loss_plot_path outputs/simple_ae_loss_curve.png
```

During training, the input representation is standardized and the fitted scaler is stored together with the trained autoencoder.

### Inference

The scaler learned from the training data must be reused during inference.

```bash
python simple_ae.py inference \
    --fused_rep_path data/fused_representation_test.csv \
    --model_load_path models/autoencoder_best.pth \
    --scaler_load_path models/scaler.pkl \
    --output_csv outputs/simple_ae_inference.csv
```

### Main Outputs

```text
autoencoder_best.pth
scaler.pkl
simple_ae.csv
```

The final CSV contains the compressed latent protein representations extracted from the autoencoder bottleneck.

---

## 4. Multimodal Transfer Autoencoder

### `transfer_ae.py`

Transfer learning from a trained three-modal autoencoder (`multi_odal_representations.py`, Sequence + Text + PPI):
its sequence encoder and its decoders initialise a sequence-only model that is trained to reconstruct all three
modalities from the sequence representation. Multimodal information can then shape the representation of proteins
for which only a sequence representation is available.

As in the multimodal autoencoder, all modalities are standardised during training; the sequence factors are saved
next to the weights (`<weights>.shift_factors.txt`, `<weights>.scaling_factors.txt`) and applied automatically in
test mode (`--no_scaling` uses the inputs as given; `--shift_factors` / `--scaling_factors` override the saved files).

### Training

```bash
python multimodal_representations/transfer_ae.py \
    --mode train \
    --seq_csv data/sequence.csv \
    --ppi_csv data/ppi.csv \
    --text_csv data/text.csv \
    --model_weights models/multi_modal_weights.pth \
    --batch_size 128 \
    --epochs 200 \
    --save_model_path models/transfer_ae_weights.pth \
    --save_csv_path outputs/transfer_representation.csv \
    --loss_plot_path outputs/transfer_ae_loss_curve.png \
    --seed 42
```

### Representation Extraction / Test Mode

```bash
python multimodal_representations/transfer_ae.py \
    --mode test \
    --seq_csv data/sequence_test.csv \
    --model_weights models/transfer_ae_weights.pth \
    --save_csv_path outputs/transfer_representation_test.csv
```

### Main Output

```text
transfer_representation.csv
```

contains sequence-derived latent representations informed by the previously learned multimodal feature space.

---

## 5. Sequence–Text Transfer Autoencoder

### `transfer_text_seq.py`

This script provides the corresponding transfer-learning architecture for the dual-modal sequence–text model.

The pretrained model initially learns a shared representation from:

```text
Sequence + Text
```

The learned parameters are subsequently transferred to a sequence-based autoencoder.

This allows latent representations to be generated using sequence information while retaining knowledge learned from the text modality during multimodal training.

### Training

```bash
python transfer_text_seq.py \
    --mode train \
    --seq_csv data/sequence.csv \
    --text_csv data/text.csv \
    --model_weights models/dual_modal_weights.pth \
    --representation_dim 512 \
    --batch_size 128 \
    --epochs 200 \
    --save_model_path models/transfer_text_seq_weights.pth \
    --save_csv_path outputs/transfer_text_seq_representation.csv \
    --loss_plot_path outputs/transfer_text_seq_loss_curve.png \
    --seed 42
```

### Representation Extraction / Test Mode

```bash
python transfer_text_seq.py \
    --mode test \
    --seq_csv data/sequence_test.csv \
    --model_weights models/transfer_text_seq_weights.pth \
    --save_csv_path outputs/transfer_text_seq_test.csv
```

The representation size is that of the dual-modal model given in `--model_weights` (`--representation_dim` is
ignored). As in the dual-modal autoencoder, sequence and text representations are standardised during training
(`--no_scaling` uses them as given); the sequence factors are saved next to the weights and applied automatically in
test mode (`--shift_factors` / `--scaling_factors` override them).

### Main Output

```text
transfer_text_seq_representation.csv
```

contains sequence-based representations transferred from the sequence–text latent space.

---

# Reproducibility

Random seeds are explicitly controlled in the training scripts to improve computational reproducibility.

The implementations set seeds for major random-number generators, including:

```text
Python random
NumPy
PyTorch
CUDA
```

The default random seed is:

```text
42
```

Alternative seeds can be specified using:

```bash
--seed <integer>
```

---

# Model Outputs

Depending on the selected workflow, the repository generates:

- trained PyTorch model weights (`.pth`),
- standardized-data scaler objects (`.pkl`),
- latent protein representations (`.csv`),
- training and validation loss curves (`.png`).

The resulting CSV representations can subsequently be incorporated into downstream supervised or unsupervised learning pipelines.

---
