"""Dual-modal (sequence + text) autoencoder; its weights initialise TransferAE (transfer_text_seq.py).

Input dimensions are taken from the CSV files (multi-column: Entry, 0, 1, ...); only proteins present in
both files are used.

    python multimodal_representations/multimodal_text_seq.py --seq_csv seq.csv --text_csv text.csv \
        --epochs 100 --save_model_path dual_modal_weights.pth --save_csv_path fused_dual.csv
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
matplotlib.use('Agg')
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


class DualModalAutoencoder(nn.Module):
    def __init__(self,
                 text_dim=3072, text_dim1=768, text_dim2=512,
                 seq_dim=1024, seq_dim1=768, seq_dim2=512,
                 zdim=512):
        super().__init__()
        self.encoder_text = nn.Sequential(
            nn.Linear(text_dim, text_dim1), nn.Tanh(),
            nn.Linear(text_dim1, text_dim2), nn.Tanh()
        )
        self.encoder_seq = nn.Sequential(
            nn.Linear(seq_dim, seq_dim1), nn.Tanh(),
            nn.Linear(seq_dim1, seq_dim2), nn.Tanh()
        )
        fused_dim = text_dim2 + seq_dim2
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
        self.decoder_seq = nn.Sequential(
            nn.Linear(seq_dim2, seq_dim1), nn.Tanh(),
            nn.Linear(seq_dim1, seq_dim), nn.Tanh()
        )

    def forward(self, x_text, x_seq):
        et = self.encoder_text(x_text)
        es = self.encoder_seq(x_seq)
        fused_in = torch.cat((et, es), dim=1)
        z = self.encoder_fuse(fused_in)
        fused_out = self.decoder_fuse(z)
        t2 = et.size(1)
        t_chunk = fused_out[:, :t2]
        s_chunk = fused_out[:, t2:]
        dt = self.decoder_text(t_chunk)
        ds = self.decoder_seq(s_chunk)
        return dt, ds, z


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dual-modal (Text + Seq) Autoencoder Training Script.")
    parser.add_argument("--seq_csv", type=str, required=True)
    parser.add_argument("--text_csv", type=str, required=True)
    parser.add_argument("--representation_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model_path", type=str, default="dual_modal_weights.pth")
    parser.add_argument("--save_csv_path", type=str, default="fused_dual.csv")
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_plot_path", type=str, default="loss_curve.png")
    parser.add_argument("--no_scaling", action="store_true",
                        help="Use the representations as given instead of standardising them")
    return parser.parse_args()


def load_and_preprocess_data(args):
    # Each modality is standardised over its whole file (as in MultiModalAE and transfer_text_seq.py); unscaled
    # representations have very small values and the training diverges after a few epochs.
    seq = rep_io.read_representation(args.seq_csv, scale=not args.no_scaling)
    text = rep_io.read_representation(args.text_csv, scale=not args.no_scaling)
    entries, (seq_v, text_v) = rep_io.align(seq, text)
    seq_t = torch.tensor(seq_v, dtype=torch.float)
    text_t = torch.tensor(text_v, dtype=torch.float)
    N = len(entries)
    idx = list(range(N))
    np.random.shuffle(idx)
    split = int(args.validation_split * N)
    val_i, train_i = idx[:split], idx[split:]
    return entries, seq_t, text_t, train_i, val_i


def plot_losses(train_hist, val_hist, path):
    plt.figure()
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(path)


def train_model(model, train_i, val_i, seq_t, text_t,
                criterion, optimizer, scheduler, batch_size, num_epochs, device):
    best_w = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_hist, val_hist = [], []
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            loader = DataLoader(train_i if phase == 'train' else val_i,
                                batch_size=batch_size, shuffle=(phase == 'train'))
            running = 0
            for bi in loader:
                bi = bi.long()
                x_s = seq_t[bi].to(device)
                x_t = text_t[bi].to(device)
                if phase == 'train': optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    dt, ds, _ = model(x_t, x_s)
                    loss = criterion(dt, x_t) + criterion(ds, x_s)
                    if phase == 'train': loss.backward(); optimizer.step()
                running += loss.item()
            epoch_loss = running / len(loader)
            if phase == 'train': train_hist.append(epoch_loss)
            else:
                val_hist.append(epoch_loss)
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss, best_w = epoch_loss, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_hist[-1]:.4f}, Val: {val_hist[-1]:.4f}")
    model.load_state_dict(best_w)
    return model, train_hist, val_hist


def extract_fused_representations(model, entries, seq_t, text_t, device, batch_size=512):
    """Latent (encoder_fuse) vectors for all proteins, as a multi-column DataFrame."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            sl = slice(start, start + batch_size)
            _, _, z = model(text_t[sl].to(device), seq_t[sl].to(device))
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

    entries, seq_t, text_t, train_i, val_i = load_and_preprocess_data(args)
    model = DualModalAutoencoder(text_dim=text_t.shape[1], seq_dim=seq_t.shape[1],
                                 zdim=args.representation_dim).to(device)

    if args.inference:
        assert args.load_model_path, "--load_model_path is required in inference mode"
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        rep_df = extract_fused_representations(model, entries, seq_t, text_t, device)
        rep_df.to_csv(args.save_csv_path, index=False)
        print(f"[Inference] Saved fused reps to {args.save_csv_path}")
        return

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    model, train_hist, val_hist = train_model(
        model, train_i, val_i, seq_t, text_t,
        criterion, optimizer, scheduler,
        args.batch_size, args.epochs, device
    )

    plot_losses(train_hist, val_hist, args.loss_plot_path)

    rep_df = extract_fused_representations(model, entries, seq_t, text_t, device)
    rep_df.to_csv(args.save_csv_path, index=False)
    print(f"Saved fused reps to {args.save_csv_path}")
    torch.save(model.state_dict(), args.save_model_path)
    print(f"Saved model weights to {args.save_model_path}")


if __name__ == "__main__":
    main()
