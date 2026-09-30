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

Usage:
  # Training + extraction:
  python multimodal_ae.py --seq_csv path/to/seq.csv --ppi_csv path/to/ppi.csv --text_csv path/to/text.csv \
       --representation_dim 512 --epochs 400 --batch_size 128 --lr 1e-3 \
       --save_model_path model.pth --save_csv_path fused.csv

  # Inference-only:
  python multimodal_ae.py --seq_csv path/to/seq.csv --ppi_csv path/to/ppi.csv --text_csv path/to/text.csv \
       --inference --load_model_path model.pth --save_csv_path fused_inference.csv
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
matplotlib.use('Agg')  # Disable GUI backends
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def convert_dataframe_to_multi_col(df: pd.DataFrame) -> pd.DataFrame:
    entry = pd.DataFrame(df['Entry'])
    vector = pd.DataFrame(list(df['Vector']))
    return pd.merge(left=entry, right=vector, left_index=True, right_index=True)


def convert_to_two_col(multi_col_df: pd.DataFrame) -> pd.DataFrame:
    vals = multi_col_df.iloc[:, 1:]
    out = pd.DataFrame(columns=['Entry', 'Vector'])
    for idx, row in tqdm.tqdm(vals.iterrows(), total=len(vals)):
        out.loc[idx] = [multi_col_df.at[idx, 'Entry'], list(row.astype(float))]
    return out


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
        #breakpoint()
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
    parser.add_argument("--seq_csv", type=str, default="T5_UNIPROT_HUMAN.csv")
    parser.add_argument("--ppi_csv", type=str, default="ppi_rep.csv")
    parser.add_argument("--text_csv", type=str, default="unipubmed_tfidf_vectors_pca1024.csv")
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
    seq = pd.read_csv(args.seq_csv,index_col=False)
    ppi = pd.read_csv(args.ppi_csv)
    text = pd.read_csv(args.text_csv)
    
    for df in (seq, ppi, text):    
        cols = df.columns.difference(['Entry'])
        #breakpoint()
        df[cols] = StandardScaler().fit_transform(df[cols])
    #breakpoint()
    seq2 = convert_to_two_col(seq)
    ppi2 = convert_to_two_col(ppi)
    text2 = convert_to_two_col(text)
    
    merged = seq2.merge(ppi2, on='Entry').merge(text2, on='Entry')
    merged.columns = ['Entry', 'sequence', 'ppi', 'text']
    seq_t = torch.tensor(list(merged['sequence']), dtype=torch.float)
    ppi_t = torch.tensor(list(merged['ppi']), dtype=torch.float)
    text_t= torch.tensor(list(merged['text']), dtype=torch.float)
    N = len(merged)
    idx = list(range(N))
    np.random.shuffle(idx)
    split = int(args.validation_split * N)
    val_i, train_i = idx[:split], idx[split:]
    return merged, seq_t, text_t, ppi_t, train_i, val_i


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
                bi = torch.tensor(bi, device=device)
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


def extract_fused_representations(model, merged, seq_t, text_t, ppi_t, device):
    activation = {}
    def hook_fn(_, _in, out): activation['z'] = out.detach()
    handle = model.encoder_fuse.register_forward_hook(hook_fn)
    model.eval()
    out_list = []
    for idx, row in tqdm.tqdm(merged.iterrows(), total=len(merged)):
        x_s = seq_t[idx].unsqueeze(0).to(device)
        x_p = ppi_t[idx].unsqueeze(0).to(device)
        x_t = text_t[idx].unsqueeze(0).to(device)
        _ = model(x_t, x_p, x_s)
        vec = activation['z'].cpu().numpy().flatten().tolist()
        out_list.append({'Entry': row['Entry'], 'Vector': vec})
    handle.remove()
    return pd.DataFrame(out_list)


def main():
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    merged, seq_t, text_t, ppi_t, train_i, val_i = load_and_preprocess_data(args)
    model = MultiModalAutoencoder(zdim=args.representation_dim).to(device)

    if args.inference:
        assert args.load_model_path, "--load_model_path is required in inference mode"
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        rep_df = extract_fused_representations(model, merged, seq_t, text_t, ppi_t, device)
        convert_dataframe_to_multi_col(rep_df).to_csv(args.save_csv_path, index=False)
        print(f"[Inference] Saved fused reps to {args.save_csv_path}")
        return

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    model, train_hist, val_hist = train_model(
        model, train_i, val_i, seq_t, text_t, ppi_t,
        criterion, optimizer, scheduler,
        args.batch_size, args.epochs, device
    )
    
    plot_losses(train_hist, val_hist, args.loss_plot_path)

    # extract and save after training
    rep_df = extract_fused_representations(model, merged, seq_t, text_t, ppi_t, device)
    convert_dataframe_to_multi_col(rep_df).to_csv(args.save_csv_path, index=False)
    print(f"Saved fused reps to {args.save_csv_path}")

    torch.save(model.state_dict(), args.save_model_path)
    print(f"Saved model weights to {args.save_model_path}")

if __name__ == "__main__":
    main()

