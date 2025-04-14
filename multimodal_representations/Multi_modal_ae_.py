"""
Multi-Modal Autoencoder Training and Representation Extraction Module

This module implements a multi-modal autoencoder that fuses three different modalities:
  - Text modality
  - Protein–Protein Interaction (PPI) modality
  - Sequence modality

It provides functionality to:
  • Load and preprocess CSV data for each modality.
  • Scale and merge representations into a unified format.
  • Build and train a multi-modal autoencoder using PyTorch.
  • Extract fused (latent) representations via a forward hook.
  • Plot training and validation loss curves.
  • Save the fused representations and model weights.

Usage:
  Run this module from the command line with appropriate arguments:
    python multimodal_ae.py --seq_csv path/to/seq.csv --ppi_csv path/to/ppi.csv --text_csv path/to/text.csv \
       --representation_dim 512 --epochs 400 --batch_size 128 --lr 1e-3
"""

import os
import random
import time
import copy
import argparse

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tqdm
import matplotlib
matplotlib.use('Agg')  # Tkinter gibi GUI backend'lerini devre dışı bırakır
import matplotlib.pyplot as plt

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def convert_dataframe_to_multi_col(representation_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame with columns ['Entry', 'Vector'] (where Vector is a list)
    into a multi-column DataFrame with one column per vector element.

    Args:
        representation_dataframe (pd.DataFrame): DataFrame with 'Entry' and 'Vector'.

    Returns:
        pd.DataFrame: DataFrame with 'Entry' and separate columns for each vector element.
    """
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    return pd.merge(left=entry, right=vector, left_index=True, right_index=True)


def convert_to_two_col(multi_col_representation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a multi-column DataFrame (with 'Entry' and numeric columns) into a
    2-column DataFrame with 'Entry' and a 'Vector' (a list of floats).

    Args:
        multi_col_representation_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Two-column DataFrame with 'Entry' and 'Vector'.
    """
    vals = multi_col_representation_df.iloc[:, 1:]
    original_values_as_df = pd.DataFrame(columns=['Entry', 'Vector'])
    for index, row in tqdm.tqdm(vals.iterrows(), total=len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [
            multi_col_representation_df.iloc[index]['Entry'], list_of_floats
        ]
    return original_values_as_df


class MultiModalAutoencoder(nn.Module):
    """
    A multi-modal autoencoder that fuses three separate inputs (text, PPI, sequence)
    into a single latent representation and reconstructs each modality.
    """
    def __init__(self, 
                 text_dim: int = 1024,
                 text_dim1: int = 768,
                 text_dim2: int = 512,
                 ppi_dim: int = 1000,
                 ppi_dim1: int = 1000,
                 ppi_dim2: int = 1000,
                 seq_dim: int = 1024,
                 seq_dim1: int = 768,
                 seq_dim2: int = 512,
                 zdim: int = 512):
        super(MultiModalAutoencoder, self).__init__()
        self.text_dim = text_dim
        self.text_dim1 = text_dim1
        self.text_dim2 = text_dim2
        self.ppi_dim = ppi_dim
        self.ppi_dim1 = ppi_dim1
        self.ppi_dim2 = ppi_dim2
        self.seq_dim = seq_dim
        self.seq_dim1 = seq_dim1
        self.seq_dim2 = seq_dim2
        self.zdim = zdim

        # Encoders
        self.encoder_text = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim1),
            nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim2),
            nn.Tanh()
        )
        self.encoder_ppi = nn.Sequential(
            nn.Linear(self.ppi_dim, self.ppi_dim1),
            nn.Tanh(),
            nn.Linear(self.ppi_dim1, self.ppi_dim2),
            nn.Tanh()
        )
        self.encoder_seq = nn.Sequential(
            nn.Linear(self.seq_dim, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim2),
            nn.Tanh()
        )

        total_fused_dim = self.text_dim2 + self.ppi_dim2 + self.seq_dim2
        self.encoder_fuse = nn.Sequential(
            nn.Linear(total_fused_dim, self.zdim),
            nn.Tanh()
        )

        # Decoders
        self.decoder_fuse = nn.Sequential(
            nn.Linear(self.zdim, total_fused_dim),
            nn.Tanh()
        )
        self.decoder_text = nn.Sequential(
            nn.Linear(self.text_dim2, self.text_dim1),
            nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim),
            nn.Tanh()
        )
        self.decoder_ppi = nn.Sequential(
            nn.Linear(self.ppi_dim2, self.ppi_dim1),
            nn.Tanh(),
            nn.Linear(self.ppi_dim1, self.ppi_dim),
            nn.Tanh()
        )
        self.decoder_seq = nn.Sequential(
            nn.Linear(self.seq_dim2, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim),
            nn.Tanh()
        )

    def forward(self, x_text, x_ppi, x_seq):
        """
        Forward pass through the autoencoder.

        Args:
            x_text (torch.Tensor): Batch of text features.
            x_ppi (torch.Tensor): Batch of PPI features.
            x_seq (torch.Tensor): Batch of sequence features.

        Returns:
            Tuple[torch.Tensor]: Decoded outputs for text, PPI, sequence, and latent vector z.
        """
        encoded_text = self.encoder_text(x_text)
        encoded_ppi = self.encoder_ppi(x_ppi)
        encoded_seq = self.encoder_seq(x_seq)
        fused_input = torch.cat((encoded_text, encoded_ppi, encoded_seq), dim=1)
        z = self.encoder_fuse(fused_input)
        fused_output = self.decoder_fuse(z)
        text_chunk = fused_output[:, :self.text_dim2]
        ppi_chunk = fused_output[:, self.text_dim2:self.text_dim2 + self.ppi_dim2]
        seq_chunk = fused_output[:, self.text_dim2 + self.ppi_dim2:]
        decoded_text = self.decoder_text(text_chunk)
        decoded_ppi = self.decoder_ppi(ppi_chunk)
        decoded_seq = self.decoder_seq(seq_chunk)
        return decoded_text, decoded_ppi, decoded_seq, z


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-modal Autoencoder Training Script.")
    parser.add_argument("--seq_csv", type=str, default="T5_UNIPROT_HUMAN.csv",
                        help="Path to sequence representation CSV.")
    parser.add_argument("--ppi_csv", type=str, default="ppi_rep.csv",
                        help="Path to PPI representation CSV.")
    parser.add_argument("--text_csv", type=str, default="unipubmed_tfidf_vectors_pca1024.csv",
                        help="Path to text representation CSV.")
    parser.add_argument("--representation_dim", type=int, default=512,
                        help="Dimension of final fused representation (z-dim).")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Fraction of dataset to use for validation.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer.")
    parser.add_argument("--save_model_path", type=str, default="multi_modal_weights.pth",
                        help="File path to save trained model weights.")
    parser.add_argument("--save_csv_path", type=str, default="fused_representation.csv",
                        help="File path to save fused representation CSV.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()


def load_and_preprocess_data(args: argparse.Namespace) -> tuple:
    """
    Load CSV data for each modality, scale features, convert to a two-column format,
    merge them into a single DataFrame, and convert merged vectors into Torch tensors.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        
    Returns:
        tuple: (fuse_phase_2 DataFrame, sequence_tensors, text_tensors, ppi_tensors, train_indices, val_indices)
    """
    # Load CSVs
    seq_rep_multi_col = pd.read_csv(args.seq_csv)
    ppi_rep_multi_col = pd.read_csv(args.ppi_csv)
    text_rep_multi_col = pd.read_csv(args.text_csv)

    # Scale data (all columns except 'Entry')
    scaler_seq = StandardScaler()
    seq_cols = seq_rep_multi_col.columns.difference(['Entry'])
    seq_rep_multi_col[seq_cols] = scaler_seq.fit_transform(seq_rep_multi_col[seq_cols])

    scaler_ppi = StandardScaler()
    ppi_cols = ppi_rep_multi_col.columns.difference(['Entry'])
    ppi_rep_multi_col[ppi_cols] = scaler_ppi.fit_transform(ppi_rep_multi_col[ppi_cols])

    scaler_text = StandardScaler()
    text_cols = text_rep_multi_col.columns.difference(['Entry'])
    text_rep_multi_col[text_cols] = scaler_text.fit_transform(text_rep_multi_col[text_cols])

    # Convert to two-column format
    sequence_rep = convert_to_two_col(seq_rep_multi_col)
    ppi_rep = convert_to_two_col(ppi_rep_multi_col)
    text_rep = convert_to_two_col(text_rep_multi_col)

    # Merge DataFrames on 'Entry'
    fuse_phase_1 = sequence_rep.merge(ppi_rep, on='Entry')
    fuse_phase_2 = fuse_phase_1.merge(text_rep, on='Entry')
    fuse_phase_2.columns = ["Entry", "sequence", "ppi", "text"]

    # Convert merged vectors to Torch tensors
    sequence_tensors = torch.tensor(list(fuse_phase_2['sequence'].values), dtype=torch.float)
    ppi_tensors = torch.tensor(list(fuse_phase_2['ppi'].values), dtype=torch.float)
    text_tensors = torch.tensor(list(fuse_phase_2['text'].values), dtype=torch.float)

    # Create train/validation indices
    dataset_size = len(sequence_tensors)
    indices = list(range(dataset_size))
    split = int(np.floor(args.validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    return fuse_phase_2, sequence_tensors, text_tensors, ppi_tensors, train_indices, val_indices


def plot_losses(train_loss_history: list, val_loss_history: list) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_loss_history (list): List of training losses per epoch.
        val_loss_history (list): List of validation losses per epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.title("Training and Validation Loss")
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def train_model(model: nn.Module,
                train_indices: list,
                val_indices: list,
                sequence_tensors: torch.Tensor,
                text_tensors: torch.Tensor,
                ppi_tensors: torch.Tensor,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: lr_scheduler._LRScheduler,
                batch_size: int,
                num_epochs: int,
                device: torch.device) -> tuple:
    """
    Train the multi-modal autoencoder using train/validation splits.

    Args:
        model (nn.Module): The PyTorch model.
        train_indices (list): Indices for training samples.
        val_indices (list): Indices for validation samples.
        sequence_tensors (torch.Tensor): Sequence modality tensor.
        text_tensors (torch.Tensor): Text modality tensor.
        ppi_tensors (torch.Tensor): PPI modality tensor.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
        batch_size (int): Batch size.
        num_epochs (int): Number of training epochs.
        device (torch.device): Training device.

    Returns:
        tuple: (best_model, train_loss_history, val_loss_history)
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    # Create DataLoaders from indices
    train_loader = DataLoader(train_indices, batch_size=batch_size,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_indices, batch_size=batch_size,
                            shuffle=False, pin_memory=True)

    seq_dim = sequence_tensors.shape[1]
    ppi_dim = ppi_tensors.shape[1]
    text_dim = text_tensors.shape[1]

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            data_loader = train_loader if phase == "train" else val_loader
            running_loss = 0.0

            for batch_indices in data_loader:
                batch_indices = batch_indices.to(device)
                seq_batch = sequence_tensors[batch_indices].view(-1, seq_dim).to(device)
                ppi_batch = ppi_tensors[batch_indices].view(-1, ppi_dim).to(device)
                text_batch = text_tensors[batch_indices].view(-1, text_dim).to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    decoded_text, decoded_ppi, decoded_seq, _ = model(text_batch, ppi_batch, seq_batch)
                    loss = (criterion(decoded_text, text_batch) +
                            criterion(decoded_ppi, ppi_batch) +
                            criterion(decoded_seq, seq_batch))
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loader)
            if phase == "train":
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss_history[-1]:.4f} "
              f"Val Loss: {val_loss_history[-1]:.4f}")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history


def extract_fused_representations(model: nn.Module,
                                  fuse_phase_2: pd.DataFrame,
                                  sequence_tensors: torch.Tensor,
                                  text_tensors: torch.Tensor,
                                  ppi_tensors: torch.Tensor,
                                  device: torch.device) -> pd.DataFrame:
    """
    Extract fused representations from the trained autoencoder using a forward hook.

    Args:
        model (nn.Module): The trained model.
        fuse_phase_2 (pd.DataFrame): Merged DataFrame containing 'Entry' and modalities.
        sequence_tensors (torch.Tensor): Sequence modality tensor.
        text_tensors (torch.Tensor): Text modality tensor.
        ppi_tensors (torch.Tensor): PPI modality tensor.
        device (torch.device): Device.

    Returns:
        pd.DataFrame: DataFrame with columns ['Entry', 'Vector'].
    """
    activation = {}

    def get_activation(name):
        def hook(_model, _input, output):
            activation[name] = output.detach()
        return hook

    hook_handle = model.encoder_fuse.register_forward_hook(get_activation('encoder_fuse'))
    model.eval()
    fused_rep_list = []
    seq_dim = sequence_tensors.shape[1]
    ppi_dim = ppi_tensors.shape[1]
    text_dim = text_tensors.shape[1]

    with torch.no_grad():
        for idx, row in tqdm.tqdm(fuse_phase_2.iterrows(), total=len(fuse_phase_2)):
            text_batch = text_tensors[idx].view(-1, text_dim).to(device)
            ppi_batch = ppi_tensors[idx].view(-1, ppi_dim).to(device)
            seq_batch = sequence_tensors[idx].view(-1, seq_dim).to(device)
            _ = model(text_batch, ppi_batch, seq_batch)
            fused_vector = activation['encoder_fuse'].cpu().numpy().tolist()[0]
            fused_rep_list.append({'Entry': row['Entry'], 'Vector': fused_vector})

    hook_handle.remove()
    return pd.DataFrame(fused_rep_list)


def main():
    """
    Main routine that orchestrates:
      1. Argument parsing and seeding.
      2. Data loading and preprocessing.
      3. Model creation, training, and evaluation.
      4. Fused representation extraction and saving results.
    """
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    (fuse_phase_2, sequence_tensors, text_tensors, ppi_tensors,
     train_indices, val_indices) = load_and_preprocess_data(args)

    # Build model, loss, optimizer, and scheduler
    model = MultiModalAutoencoder(zdim=args.representation_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    # Train model
    best_model, train_loss_history, val_loss_history = train_model(
        model=model,
        train_indices=train_indices,
        val_indices=val_indices,
        sequence_tensors=sequence_tensors,
        text_tensors=text_tensors,
        ppi_tensors=ppi_tensors,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=device
    )

    # Plot training losses
    plot_losses(train_loss_history, val_loss_history)

    # Extract fused representations
    fused_rep_df = extract_fused_representations(
        best_model, fuse_phase_2, sequence_tensors, text_tensors, ppi_tensors, device
    )
    fused_rep_multi_col = convert_dataframe_to_multi_col(fused_rep_df)
    fused_rep_multi_col.to_csv(args.save_csv_path, index=False)
    print(f"Fused representation saved at {args.save_csv_path}")

    # Save model weights
    torch.save(best_model.state_dict(), args.save_model_path)
    print(f"Model weights saved at {args.save_model_path}")


if __name__ == "__main__":
    main()
