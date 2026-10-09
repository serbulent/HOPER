"""
Multi-Modal Autoencoder Training, Inference and Representation Extraction Module

This module implements a multi-modal autoencoder that fuses three different modalities:
  - Text modality
  - Protein–Protein Interaction (PPI) modality
  - Sequence modality

It provides functionality to:
  • Load and preprocess CSV data for each modality.
  • Scale and merge representations into a unified format.
  • Build, train, and/or load a multi-modal autoencoder using PyTorch.
  • Extract fused (latent) representations via a forward hook.
  • Plot training and validation loss curves.
  • Save fused representations and model weights.

The input dimensions are taken from the CSV files (multi-column: Entry, 0, 1, ...); only proteins present
in all three files are used.

Usage:
  # Training + extraction:
  python multimodal_representations/multi_odal_representations.py --seq_csv seq.csv --ppi_csv ppi.csv \
       --text_csv text.csv --representation_dim 512 --epochs 100 --batch_size 128 --lr 1e-3 \
       --save_model_path model.pth --save_csv_path fused.csv

  # Inference-only (same input dimensions as the trained model):
  python multimodal_representations/multi_odal_representations.py --seq_csv seq.csv --ppi_csv ppi.csv \
       --text_csv text.csv --inference --load_model_path model.pth --save_csv_path fused_inference.csv
"""

import os
import sys
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
import matplotlib
matplotlib.use('Agg')  # Disable GUI backends
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rep_io


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class MultiModalAutoencoder(nn.Module):
    def __init__(self,
                 text_dim=3072, text_dim1=768, text_dim2=512,
                 ppi_dim=500, ppi_dim1=1000, ppi_dim2=1000,
                 seq_dim=1024, seq_dim1=768, seq_dim2=512,
                 zdim=512):
        super().__init__()
        self.encoder_text = nn.Sequential(
            nn.Linear(text_dim, text_dim1), nn.Tanh(),
            nn.Linear(text_dim1, text_dim2), nn.Tanh()
        )
        self.encoder_ppi = nn.Sequential(
            nn.Linear(ppi_dim, ppi_dim1), nn.Tanh(),
            nn.Linear(ppi_dim1, ppi_dim2), nn.Tanh()
        )
        self.encoder_seq = nn.Sequential(
            nn.Linear(seq_dim, seq_dim1), nn.Tanh(),
            nn.Linear(seq_dim1, seq_dim2), nn.Tanh()
        )
        fused_dim = text_dim2 + ppi_dim2 + seq_dim2
        self.encoder_fuse = nn.Sequential(
            nn.Linear(fused_dim, zdim), nn.Tanh()
        )
        self.decoder_fuse = nn.Sequential(
            nn.Linear(zdim, fused_dim), nn.Tanh()
        )
        self.decoder_text = nn.Sequential(
            nn.Linear(text_dim2, text_dim1), nn.Tanh(),
            nn.Linear(text_dim1, text_dim), nn.Tanh()
        )
        self.decoder_ppi = nn.Sequential(
            nn.Linear(ppi_dim2, ppi_dim1), nn.Tanh(),
            nn.Linear(ppi_dim1, ppi_dim), nn.Tanh()
        )
        self.decoder_seq = nn.Sequential(
            nn.Linear(seq_dim2, seq_dim1), nn.Tanh(),
            nn.Linear(seq_dim1, seq_dim), nn.Tanh()
        )

    def forward(self, x_text, x_ppi, x_seq):
        et = self.encoder_text(x_text)
        ep = self.encoder_ppi(x_ppi)
        es = self.encoder_seq(x_seq)
        fused_in = torch.cat((et, ep, es), dim=1)
        z = self.encoder_fuse(fused_in)
        fused_out = self.decoder_fuse(z)
        t2 = et.size(1)
        p2 = ep.size(1)
        # split fused_out
        t_chunk = fused_out[:, :t2]
        p_chunk = fused_out[:, t2:t2+p2]
        s_chunk = fused_out[:, t2+p2:]
        dt = self.decoder_text(t_chunk)
        dp = self.decoder_ppi(p_chunk)
        ds = self.decoder_seq(s_chunk)
        return dt, dp, ds, z


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-modal Autoencoder Training Script.")
    parser.add_argument("--seq_csv", type=str, required=True, help="Sequence representations (e.g. ProtT5)")
    parser.add_argument("--ppi_csv", type=str, required=True, help="PPI representations (e.g. Node2vec)")
    parser.add_argument("--text_csv", type=str, required=True, help="Text representations (e.g. TF-IDF SVD)")
    parser.add_argument("--representation_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model_path", type=str, default="multi_modal_weights.pth")
    parser.add_argument("--save_csv_path", type=str, default="fused_representation.csv")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="(Inference) Path to pretrained model weights.")
    parser.add_argument("--inference", action="store_true",
                        help="Run in inference-only mode (skip training).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_plot_path", type=str, default="loss_curve.png",
                        help="Path to save the training/validation loss curve image.")
    return parser.parse_args()


def load_and_preprocess_data(args):
    # Each modality is standardised separately, then the proteins common to all three are kept.
    seq = rep_io.read_representation(args.seq_csv, scale=True)
    ppi = rep_io.read_representation(args.ppi_csv, scale=True)
    text = rep_io.read_representation(args.text_csv, scale=True)
    entries, (seq_v, ppi_v, text_v) = rep_io.align(seq, ppi, text)
    seq_t = torch.tensor(seq_v, dtype=torch.float)
    ppi_t = torch.tensor(ppi_v, dtype=torch.float)
    text_t = torch.tensor(text_v, dtype=torch.float)
    N = len(entries)
    idx = list(range(N))
    np.random.shuffle(idx)
    split = int(args.validation_split * N)
    val_i, train_i = idx[:split], idx[split:]
    return entries, seq_t, text_t, ppi_t, train_i, val_i


def plot_losses(train_hist, val_hist,save_path):
    plt.figure()
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(save_path)


def train_model(model, train_i, val_i, seq_t, text_t, ppi_t,
                criterion, optimizer, scheduler, batch_size, num_epochs, device):
    best_w = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_hist, val_hist = [], []
    
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            loader = DataLoader(train_i if phase=='train' else val_i,batch_size=batch_size, shuffle=(phase=='train'))
            running=0
            for bi in loader:
                bi = bi.long()
                x_s = seq_t[bi].to(device); x_p = ppi_t[bi].to(device); x_t = text_t[bi].to(device)
                if phase=='train': optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    dt, dp, ds, _ = model(x_t, x_p, x_s)
                    loss = criterion(dt, x_t) + criterion(dp, x_p) + criterion(ds, x_s)
                    if phase=='train': loss.backward(); optimizer.step()
                running += loss.item()
            epoch_loss = running/len(loader)
            if phase=='train': train_hist.append(epoch_loss)
            else:
                val_hist.append(epoch_loss)
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss, best_w = epoch_loss, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_hist[-1]:.4f}, Val: {val_hist[-1]:.4f}")
    model.load_state_dict(best_w)
    return model, train_hist, val_hist


def extract_fused_representations(model, entries, seq_t, text_t, ppi_t, device, batch_size=512):
    """Latent (encoder_fuse) vectors for all proteins, as a multi-column DataFrame."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            sl = slice(start, start + batch_size)
            _, _, _, z = model(text_t[sl].to(device), ppi_t[sl].to(device), seq_t[sl].to(device))
            chunks.append(z.cpu().numpy())
    return rep_io.to_multi_col(entries, np.concatenate(chunks))


def ensure_parent_dirs(*paths):
    for path in paths:
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device(os.environ.get('HOPER_DEVICE', 'cpu'))  # set HOPER_DEVICE=cuda for GPU
    print(f"Using device: {device}")
    ensure_parent_dirs(args.save_model_path, args.save_csv_path, args.loss_plot_path)

    entries, seq_t, text_t, ppi_t, train_i, val_i = load_and_preprocess_data(args)
    model = MultiModalAutoencoder(text_dim=text_t.shape[1], ppi_dim=ppi_t.shape[1], seq_dim=seq_t.shape[1],
                                  zdim=args.representation_dim).to(device)

    if args.inference:
        assert args.load_model_path, "--load_model_path is required in inference mode"
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        rep_df = extract_fused_representations(model, entries, seq_t, text_t, ppi_t, device)
        rep_df.to_csv(args.save_csv_path, index=False)
        print(f"[Inference] Saved fused reps to {args.save_csv_path}")
        return

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    model, train_hist, val_hist = train_model(
        model, train_i, val_i, seq_t, text_t, ppi_t,
        criterion, optimizer, scheduler,
        args.batch_size, args.epochs, device
    )
    
    plot_losses(train_hist, val_hist, args.loss_plot_path)

    # extract and save after training
    rep_df = extract_fused_representations(model, entries, seq_t, text_t, ppi_t, device)
    rep_df.to_csv(args.save_csv_path, index=False)
    print(f"Saved fused reps to {args.save_csv_path}")

    torch.save(model.state_dict(), args.save_model_path)
    print(f"Saved model weights to {args.save_model_path}")

if __name__ == "__main__":
    main()

