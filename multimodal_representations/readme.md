
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
| `Transfer_ae.py` | Transfer Autoencoder | Sequence + pretrained multimodal model | Transferring information learned from Sequence + Text + PPI |
| `transfer_text_seq.py` | Transfer Autoencoder | Sequence + pretrained dual-modal model | Transferring information learned from Sequence + Text |

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

The implementation requires Python and the following major libraries:

```text
Python
PyTorch
NumPy
pandas
scikit-learn
matplotlib
tqdm
```

Installation can be performed using:

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
```

GPU acceleration is automatically used when a CUDA-enabled device is available.

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

The `Entry` field is used to match proteins across the different representation modalities.

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

### `Transfer_ae.py`

This script implements transfer learning from a previously trained three-modal autoencoder.

The original model is trained using:

```text
Sequence + Text + PPI
```

and information learned from the multimodal architecture is transferred to a sequence-based model.

This strategy enables multimodal information to influence the final representation even when only sequence information is available during representation extraction.

### Training

Training requires sequence, PPI, and text representations together with the weights of the previously trained multimodal model.

```bash
python Transfer_ae.py \
    --mode train \
    --seq_csv data/sequence.csv \
    --ppi_csv data/ppi.csv \
    --text_csv data/text.csv \
    --model_weights models/multi_modal_weights.pth \
    --representation_dim 512 \
    --batch_size 128 \
    --epochs 200 \
    --save_model_path models/transfer_ae_weights.pth \
    --save_csv_path outputs/transfer_representation.csv \
    --loss_plot_path outputs/transfer_ae_loss_curve.png \
    --seed 42
```

### Representation Extraction / Test Mode

After training, sequence-only representations can be generated using:

```bash
python Transfer_ae.py \
    --mode test \
    --seq_csv data/sequence_test.csv \
    --model_weights models/transfer_ae_weights.pth \
    --representation_dim 512 \
    --batch_size 128 \
    --save_csv_path outputs/transfer_representation_test.csv \
    --seed 42
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
    --representation_dim 512 \
    --batch_size 128 \
    --save_csv_path outputs/transfer_text_seq_test.csv \
    --seed 42
```

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
