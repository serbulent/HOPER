"""TransferAE (sequence + PPI + text -> sequence only).

A sequence-only autoencoder initialised from a trained MultiModalAE (multi_odal_representations.py): its sequence
encoder and its text / PPI / sequence decoders are copied, and the model is trained to reconstruct all three
modalities from the sequence representation alone. In test mode it produces representations for proteins that
only have a sequence representation.

Dimensions are read from the MultiModalAE weights. As in MultiModalAE, every modality is standardised over its
whole file; the sequence scaling factors are saved next to the TransferAE weights (<weights>.shift_factors.txt /
<weights>.scaling_factors.txt, applied as (x + shift) * scale) and reused in test mode.

    python multimodal_representations/transfer_ae.py --mode train --seq_csv seq.csv --ppi_csv ppi.csv \
        --text_csv text.csv --model_weights multimodal_ae_weights.pth --save_model_path transfer_ae_weights.pth \
        --save_csv_path transfer_ae_representation.csv
    python multimodal_representations/transfer_ae.py --mode test --seq_csv new_seq.csv \
        --model_weights transfer_ae_weights.pth --save_csv_path new_representation.csv
"""
import argparse
import copy
import os
import random
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rep_io


def multimodal_dims_from_state(state_dict, prefix=""):
    """Layer sizes of a MultiModalAE (text + PPI + sequence), read from its saved weights."""
    w = lambda name: state_dict[prefix + name + ".weight"].shape  # (out_features, in_features)
    return dict(text_dim=w("encoder_text.0")[1], text_dim1=w("encoder_text.0")[0], text_dim2=w("encoder_text.2")[0],
                ppi_dim=w("encoder_ppi.0")[1], ppi_dim1=w("encoder_ppi.0")[0], ppi_dim2=w("encoder_ppi.2")[0],
                seq_dim=w("encoder_seq.0")[1], seq_dim1=w("encoder_seq.0")[0], seq_dim2=w("encoder_seq.2")[0],
                zdim=w("encoder_fuse.0")[0])


class Autoencoder(nn.Module):
    """
    A multimodal autoencoder that fuses text, PPI, and sequence representations (same layers as
    MultiModalAutoencoder in multi_odal_representations.py, so its weights can be loaded).
    """

    def __init__(self,
                 text_dim: int = 3072,
                 text_dim1: int = 768,
                 text_dim2: int = 512,
                 ppi_dim: int = 500,
                 ppi_dim1: int = 1000,
                 ppi_dim2: int = 1000,
                 seq_dim: int = 1024,
                 seq_dim1: int = 768,
                 seq_dim2: int = 512,
                 zdim: int = 512):
        super(Autoencoder, self).__init__()
        self.text_dim = text_dim
        self.text_dim2 = text_dim2
        self.ppi_dim = ppi_dim
        self.ppi_dim2 = ppi_dim2
        self.seq_dim = seq_dim
        self.zdim = zdim

        # Encoders
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

        total_fused_dim = text_dim2 + ppi_dim2 + seq_dim2
        self.encoder_fuse = nn.Sequential(
            nn.Linear(total_fused_dim, zdim), nn.Tanh()
        )

        # Decoders
        self.decoder_fuse = nn.Sequential(
            nn.Linear(zdim, total_fused_dim), nn.Tanh()
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
        encoded_text = self.encoder_text(x_text)
        encoded_ppi = self.encoder_ppi(x_ppi)
        encoded_seq = self.encoder_seq(x_seq)
        fused_input = torch.cat((encoded_text, encoded_ppi, encoded_seq), dim=1)
        z = self.encoder_fuse(fused_input)
        fused_output = self.decoder_fuse(z)
        text_chunk = fused_output[:, :self.text_dim2]
        ppi_chunk = fused_output[:, self.text_dim2:self.text_dim2 + self.ppi_dim2]
        seq_chunk = fused_output[:, self.text_dim2 + self.ppi_dim2:]
        return self.decoder_text(text_chunk), self.decoder_ppi(ppi_chunk), self.decoder_seq(seq_chunk), z


class Autoencoder_Seq(nn.Module):
    """
    A sequence-only autoencoder initialised from a pretrained multimodal model: it encodes the sequence
    representation and reconstructs text, PPI and sequence representations from the bottleneck (encoder4).
    """
    def __init__(self, pretrained_model):
        super(Autoencoder_Seq, self).__init__()
        self.pretrained_model = pretrained_model
        # Layer sizes follow the pretrained multimodal autoencoder.
        self.text_dim = pretrained_model.encoder_text[0].in_features
        self.text_dim1 = pretrained_model.encoder_text[0].out_features
        self.text_dim2 = pretrained_model.encoder_text[2].out_features
        self.ppi_dim = pretrained_model.encoder_ppi[0].in_features
        self.ppi_dim1 = pretrained_model.encoder_ppi[0].out_features
        self.ppi_dim2 = pretrained_model.encoder_ppi[2].out_features
        self.seq_dim = pretrained_model.encoder_seq[0].in_features
        self.seq_dim1 = pretrained_model.encoder_seq[0].out_features
        self.seq_dim2 = pretrained_model.encoder_seq[2].out_features
        self.zdim = pretrained_model.encoder_fuse[0].out_features

        # Sequence encoder
        self.encoder3 = nn.Sequential(
            nn.Linear(self.seq_dim, self.seq_dim1), nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim2), nn.Tanh()
        )
        # Bottleneck (sequence modality only)
        self.encoder4 = nn.Sequential(
            nn.Linear(self.seq_dim2, self.zdim), nn.Tanh()
        )
        # Common decoder (shared across modalities)
        self.decoder4 = nn.Sequential(
            nn.Linear(self.zdim, self.text_dim2 + self.ppi_dim2 + self.seq_dim2), nn.Tanh()
        )
        # Text decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(self.text_dim2, self.text_dim1), nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim), nn.Tanh()
        )
        # PPI decoder
        self.decoder2 = nn.Sequential(
            nn.Linear(self.ppi_dim2, self.ppi_dim1), nn.Tanh(),
            nn.Linear(self.ppi_dim1, self.ppi_dim), nn.Tanh()
        )
        # Sequence decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(self.seq_dim2, self.seq_dim1), nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim), nn.Tanh()
        )

    def load_parameters(self):
        """Copy the sequence encoder and the decoders from the pretrained multimodal model."""
        pairs = [(self.encoder3, self.pretrained_model.encoder_seq, (0, 2)),
                 (self.decoder4, self.pretrained_model.decoder_fuse, (0,)),
                 (self.decoder3, self.pretrained_model.decoder_text, (0, 2)),
                 (self.decoder2, self.pretrained_model.decoder_ppi, (0, 2)),
                 (self.decoder1, self.pretrained_model.decoder_seq, (0, 2))]
        for target, source, layers in pairs:
            for i in layers:
                target[i].weight.data = copy.deepcopy(source[i].weight.data)
                target[i].bias.data = copy.deepcopy(source[i].bias.data)

    def forward(self, x_seq):
        encoded_seq = self.encoder3(x_seq)
        encoded_mid = self.encoder4(encoded_seq)
        decoded_mid = self.decoder4(encoded_mid)
        decoded_text = self.decoder3(decoded_mid[:, 0:self.text_dim2])
        decoded_ppi = self.decoder2(decoded_mid[:, self.text_dim2:self.text_dim2 + self.ppi_dim2])
        decoded_seq = self.decoder1(decoded_mid[:, self.text_dim2 + self.ppi_dim2:])
        return decoded_text, decoded_ppi, decoded_seq, encoded_mid


###############################################################################
# Training and Utility Functions
###############################################################################

def train_model_for_seq(model, train_loader, validation_loader, criterion, optimizer,
                        num_epochs, sequence_tensors, text_tensors, ppi_tensors, device):
    """
    Train the sequence autoencoder: the reconstruction loss (MSE) is summed over the three modalities.
    As in the original implementation, the weights after the last epoch are kept.
    """
    since = time.time()
    train_loss_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs"):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = validation_loader

            epoch_loss = 0.0
            for batch_indices in data_loader:
                sequence_batch = sequence_tensors[batch_indices].to(device)
                ppi_batch = ppi_tensors[batch_indices].to(device)
                text_batch = text_tensors[batch_indices].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    decoded_text, decoded_ppi, decoded_seq, _ = model(sequence_batch)
                    loss = (criterion(decoded_text, text_batch) +
                            criterion(decoded_ppi, ppi_batch) +
                            criterion(decoded_seq, sequence_batch))
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= max(len(data_loader), 1)

            if phase == 'val':
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)
            else:
                train_loss_history.append(epoch_loss)
        tqdm.tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss_history[-1]:.4f}, Val: {val_loss_history[-1]:.4f}")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Final val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history


def create_dataloaders(dataset_size, batch_size=128, validation_split=0.2, seed=42):
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_loader = DataLoader(train_indices, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_indices, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader


def extract_fused_representation(model, sequence_tensors, entries, device, batch_size=512):
    """Bottleneck (encoder4) vectors of all proteins, as a multi-column DataFrame."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            _, _, _, encoded = model(sequence_tensors[start:start + batch_size].to(device))
            chunks.append(encoded.cpu().numpy())
    return rep_io.to_multi_col(entries, np.concatenate(chunks))


###############################################################################
# Data
###############################################################################

def prepare_multimodal_data(seq_csv, text_csv, ppi_csv, scale):
    """Like MultiModalAE: each file is standardised over all its proteins, then the common proteins are kept.
    The sequence factors are returned so that test-mode inputs can be transformed identically."""
    seq = rep_io.read_representation(seq_csv)
    ppi = rep_io.read_representation(ppi_csv, scale=scale)
    text = rep_io.read_representation(text_csv, scale=scale)
    seq_shift = seq_scale = None
    if scale:
        seq_shift, seq_scale = rep_io.standardisation_factors(seq)
        seq = rep_io.apply_factors(seq, seq_shift, seq_scale)
    entries, (seq_v, ppi_v, text_v) = rep_io.align(seq, ppi, text)
    as_tensor = lambda v: torch.tensor(v, dtype=torch.float)
    return entries, as_tensor(seq_v), as_tensor(ppi_v), as_tensor(text_v), seq_shift, seq_scale


###############################################################################
# Main
###############################################################################

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TransferAE (sequence + PPI + text -> sequence) training and inference")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="'train' to train the model, 'test' to extract representations only.")
    parser.add_argument("--seq_csv", type=str, required=True, help="Sequence representations (multi-column CSV)")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="train: MultiModalAE weights; test: TransferAE weights")
    parser.add_argument("--save_csv_path", type=str, required=True, help="Output representation CSV")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    # train only
    parser.add_argument("--ppi_csv", type=str, help="PPI representations (train)")
    parser.add_argument("--text_csv", type=str, help="Text representations (train)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model_path", type=str, default="transfer_ae_weights.pth")
    parser.add_argument("--loss_plot_path", type=str, default="transfer_ae_loss.png")
    parser.add_argument("--no_scaling", action="store_true",
                        help="Use the representations as given instead of standardising them")
    # test only: override the scaling factors saved with the weights
    parser.add_argument("--shift_factors", type=str, default=None)
    parser.add_argument("--scaling_factors", type=str, default=None)
    return parser.parse_args()


def plot_losses(train_hist, val_hist, save_path):
    plt.figure()
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(save_path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def check_dim(name, actual, expected, weights):
    if actual != expected:
        raise SystemExit("{} representations have {} dimensions but the model in {} expects {}.".format(
            name, actual, weights, expected))


def main():
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device(os.environ.get("HOPER_DEVICE", "cpu"))  # set HOPER_DEVICE=cuda for GPU
    print(f"Using device: {device}")
    for path in (args.save_csv_path, args.save_model_path, args.loss_plot_path):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    if args.mode == "train":
        if not (args.ppi_csv and args.text_csv):
            raise SystemExit("Train mode requires --ppi_csv and --text_csv.")
        entries, seq_t, ppi_t, text_t, seq_shift, seq_scale = prepare_multimodal_data(
            args.seq_csv, args.text_csv, args.ppi_csv, scale=not args.no_scaling)

        multimodal_state = torch.load(args.model_weights, map_location=device)
        dims = multimodal_dims_from_state(multimodal_state)
        check_dim("Sequence", seq_t.shape[1], dims["seq_dim"], args.model_weights)
        check_dim("PPI", ppi_t.shape[1], dims["ppi_dim"], args.model_weights)
        check_dim("Text", text_t.shape[1], dims["text_dim"], args.model_weights)

        train_loader, validation_loader = create_dataloaders(len(seq_t), batch_size=args.batch_size, seed=args.seed)
        model = Autoencoder(**dims).to(device)
        model.load_state_dict(multimodal_state)
        seq_model = Autoencoder_Seq(model).to(device)
        seq_model.load_parameters()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(seq_model.parameters(), lr=args.lr)
        trained_model, train_hist, val_hist = train_model_for_seq(
            seq_model, train_loader, validation_loader, criterion, optimizer, args.epochs,
            seq_t, text_t, ppi_t, device)
        plot_losses(train_hist, val_hist, args.loss_plot_path)

        torch.save(trained_model.state_dict(), args.save_model_path)
        shift_path, scale_path = rep_io.factor_paths(args.save_model_path)
        for path in (shift_path, scale_path):
            if os.path.exists(path):
                os.remove(path)
        if seq_shift is not None:
            rep_io.write_factors(shift_path, seq_shift)
            rep_io.write_factors(scale_path, seq_scale)
        print(f"Trained model saved to {args.save_model_path}")

        fused_rep_df = extract_fused_representation(trained_model, seq_t, entries, device)
        fused_rep_df.to_csv(args.save_csv_path, index=False)
        print(f"Fused representations saved to {args.save_csv_path}")

    else:  # test
        seq = rep_io.read_representation(args.seq_csv)
        values = seq.values.astype(np.float64)
        saved_shift, saved_scale = rep_io.factor_paths(args.model_weights)
        shift_file = args.shift_factors or (saved_shift if os.path.exists(saved_shift) else None)
        scale_file = args.scaling_factors or (saved_scale if os.path.exists(saved_scale) else None)
        if shift_file:
            values = values + rep_io.read_factors(shift_file)
        if scale_file:
            values = values * rep_io.read_factors(scale_file)
        print("Sequence normalisation: {}".format("(x + shift) * scale" if shift_file or scale_file else "none"))
        seq_t = torch.tensor(values, dtype=torch.float)

        state = torch.load(args.model_weights, map_location=device)
        dims = multimodal_dims_from_state(state, prefix="pretrained_model.")
        check_dim("Sequence", seq_t.shape[1], dims["seq_dim"], args.model_weights)
        model = Autoencoder_Seq(Autoencoder(**dims))
        model.load_state_dict(state)
        model = model.to(device)

        fused_rep_df = extract_fused_representation(model, seq_t, list(seq.index), device)
        fused_rep_df.to_csv(args.save_csv_path, index=False)
        print(f"Fused representations saved to {args.save_csv_path}")


if __name__ == "__main__":
    main()
