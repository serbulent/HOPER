"""TransferAE: a sequence-only encoder initialised from the dual (sequence + text) autoencoder.

Train it on proteins that have both representations; then (test mode) it produces representations for
proteins that only have a sequence representation. Dimensions are taken from the data and the weights.

    python multimodal_representations/transfer_text_seq.py --mode train --seq_csv seq.csv --text_csv text.csv \
        --model_weights dual_modal_weights.pth --save_model_path transfer_ae_weights.pth --save_csv_path out.csv
    python multimodal_representations/transfer_text_seq.py --mode test --seq_csv new_seq.csv \
        --model_weights transfer_ae_weights.pth --save_csv_path new_out.csv
"""
import argparse
import os
import sys
import random
import copy
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rep_io


def dual_dims_from_state(state_dict, prefix=""):
    """Layer sizes of a dual (text + sequence) autoencoder, read from its saved weights."""
    w = lambda name: state_dict[prefix + name + ".weight"].shape  # (out_features, in_features)
    return dict(text_dim=w("encoder_text.0")[1], text_dim1=w("encoder_text.0")[0], text_dim2=w("encoder_text.2")[0],
                seq_dim=w("encoder_seq.0")[1], seq_dim1=w("encoder_seq.0")[0], seq_dim2=w("encoder_seq.2")[0],
                zdim=w("encoder_fuse.0")[0])


class Autoencoder(nn.Module):
    """
    A multimodal autoencoder that fuses text,and sequence representations.
    The model contains three separate encoders for text, and sequence inputs,
    which are then concatenated and passed through a bottleneck layer (encoder4).
    The decoder reconstructs each modality from the bottleneck representation.
    """
    
    def __init__(self, 
                 text_dim: int = 3072,
                 text_dim1: int = 768,
                 text_dim2: int = 512,
                 seq_dim: int = 1024,
                 seq_dim1: int = 768,
                 seq_dim2: int = 512,
                 zdim: int = 512):
        super(Autoencoder, self).__init__()
        self.text_dim = text_dim
        self.text_dim2 = text_dim2
        self.seq_dim = seq_dim
        self.zdim = zdim


        # Encoders
        self.encoder_text = nn.Sequential(
            nn.Linear(text_dim, text_dim1), nn.Tanh(),
            nn.Linear(text_dim1, text_dim2), nn.Tanh()
        )
        
        self.encoder_seq = nn.Sequential(
            nn.Linear(seq_dim, seq_dim1), nn.Tanh(),
            nn.Linear(seq_dim1, seq_dim2), nn.Tanh()
        )

        total_fused_dim = text_dim2  + seq_dim2
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
        
        self.decoder_seq = nn.Sequential(
            nn.Linear(seq_dim2, seq_dim1), nn.Tanh(),
            nn.Linear(seq_dim1, seq_dim), nn.Tanh()
        )

    def forward(self, x_text,  x_seq):
        encoded_text = self.encoder_text(x_text)
        encoded_seq = self.encoder_seq(x_seq)
        fused_input = torch.cat((encoded_text, encoded_seq), dim=1)
        z = self.encoder_fuse(fused_input)
        fused_output = self.decoder_fuse(z)
        text_chunk = fused_output[:, :self.text_dim2]
        
        seq_chunk = fused_output[:, self.text_dim2:]
        decoded_text = self.decoder_text(text_chunk)
        
        decoded_seq = self.decoder_seq(seq_chunk)
        return decoded_text, decoded_seq, z


class Autoencoder_Seq(nn.Module):
    """
    A specialized autoencoder for sequence data that leverages a pretrained multimodal model.
    
    This model uses the pretrained weights from a full multimodal autoencoder (passed as
    the `pretrained_model` argument) to initialize its sequence encoder and the common decoder.
    It only processes sequence input, then reconstructs text, PPI, and sequence outputs.
    """
    def __init__(self, pretrained_model):
        super(Autoencoder_Seq, self).__init__()
        self.pretrained_model = pretrained_model
        # Layer sizes follow the pretrained dual autoencoder.
        self.text_dim = pretrained_model.encoder_text[0].in_features
        self.text_dim1 = pretrained_model.encoder_text[0].out_features
        self.text_dim2 = pretrained_model.encoder_text[2].out_features

        self.seq_dim = pretrained_model.encoder_seq[0].in_features
        self.seq_dim1 = pretrained_model.encoder_seq[0].out_features
        self.seq_dim2 = pretrained_model.encoder_seq[2].out_features
        self.zdim = pretrained_model.encoder_fuse[0].out_features

        # Sequence encoder
        self.encoder3 = nn.Sequential(
            nn.Linear(self.seq_dim, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim2),
            nn.Tanh()
        )
        # Bottleneck (fusing only sequence modality)
        self.encoder4 = nn.Sequential(
            nn.Linear(self.seq_dim2, self.zdim),
            nn.Tanh()
        )
        # Common decoder (shared across modalities)
        self.decoder4 = nn.Sequential(
            nn.Linear(self.zdim, self.text_dim2 + self.seq_dim2),
            nn.Tanh()
        )
        # Text decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(self.text_dim2, self.text_dim1),
            nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim),
            nn.Tanh()
        )
        
        # Sequence decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(self.seq_dim2, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim),
            nn.Tanh()
        )
        
    def load_parameters(self):
        """
        Load parameters from the pretrained multimodal model into the sequence autoencoder.
        
        This copies weights from specific layers of the pretrained model into this model's layers.
        """
      
        self.encoder3[0].weight.data = copy.deepcopy(self.pretrained_model.encoder_seq[0].weight.data)
        self.encoder3[2].weight.data = copy.deepcopy(self.pretrained_model.encoder_seq[2].weight.data)
        self.encoder3[0].bias.data = copy.deepcopy(self.pretrained_model.encoder_seq[0].bias.data)
        self.encoder3[2].bias.data = copy.deepcopy(self.pretrained_model.encoder_seq[2].bias.data)
    
    # Fused decoder için (Autoencoder'da 'decoder_fuse' olarak tanımlı)
        self.decoder4[0].weight.data = copy.deepcopy(self.pretrained_model.decoder_fuse[0].weight.data)
        self.decoder4[0].bias.data = copy.deepcopy(self.pretrained_model.decoder_fuse[0].bias.data)
    
    # Text decoder (Autoencoder'da 'decoder_text' olarak tanımlı)
        self.decoder3[0].weight.data = copy.deepcopy(self.pretrained_model.decoder_text[0].weight.data)
        self.decoder3[2].weight.data = copy.deepcopy(self.pretrained_model.decoder_text[2].weight.data)
        self.decoder3[0].bias.data = copy.deepcopy(self.pretrained_model.decoder_text[0].bias.data)
        self.decoder3[2].bias.data = copy.deepcopy(self.pretrained_model.decoder_text[2].bias.data)
    
    
    # Sequence decoder (Autoencoder'da 'decoder_seq' olarak tanımlı)
        self.decoder1[0].weight.data = copy.deepcopy(self.pretrained_model.decoder_seq[0].weight.data)
        self.decoder1[2].weight.data = copy.deepcopy(self.pretrained_model.decoder_seq[2].weight.data)
        self.decoder1[0].bias.data = copy.deepcopy(self.pretrained_model.decoder_seq[0].bias.data)
        self.decoder1[2].bias.data = copy.deepcopy(self.pretrained_model.decoder_seq[2].bias.data)


    def forward(self, x_seq):
        """
        Forward pass for the sequence-only autoencoder.
        
        Args:
            x_seq (Tensor): Input tensor for sequence features.
        
        Returns:
            Tuple: Decoded text, decoded PPI, decoded sequence, and the encoded (fused) bottleneck representation.
        """
        encoded_seq = self.encoder3(x_seq)
        encoded_mid = self.encoder4(encoded_seq)
        decoded_mid = self.decoder4(encoded_mid)
        decoded_text = self.decoder3(decoded_mid[:, 0:self.text_dim2])   
        decoded_seq = self.decoder1(decoded_mid[:, self.text_dim2: ])
        return decoded_text, decoded_seq, encoded_mid


##############################################################################
# Training and Utility Functions
###############################################################################

def train_model_for_seq(model, train_loader, validation_loader, criterion, optimizer,
                        num_epochs, sequence_tensors, text_tensors, device):
    """
    Train the sequence autoencoder model with a training and validation phase.

    This function iterates over a specified number of epochs. For each epoch,
    it processes mini-batches from both training and validation sets, computes the
    reconstruction loss (MSE) across the three modalities, and applies gradient updates
    during the training phase. A learning rate scheduler is used to reduce the learning
    rate when the validation loss plateaus.

    Args:
        model (nn.Module): The autoencoder model to be trained.
        train_loader (DataLoader): DataLoader providing training batch indices.
        validation_loader (DataLoader): DataLoader providing validation batch indices.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        optimizer (Optimizer): Optimizer (e.g., AdamW).
        num_epochs (int): Number of training epochs.
        sequence_tensors (Tensor): Tensor containing sequence representations.
        text_tensors (Tensor): Tensor containing text representations.
        ppi_tensors (Tensor): Tensor containing PPI representations.
        device (torch.device): The device to run training on.

    Returns:
        Tuple: Trained model, training loss history, validation loss history, and a list of loss values.
    """
    since = time.time()
    train_loss_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_vals = []

    # Get tensor sizes
    sequence_tensors_size = sequence_tensors.shape[1]
    
    text_tensors_size = text_tensors.shape[1]

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
                # Get mini-batch tensors and move to device
                sequence_batch = sequence_tensors[batch_indices].view(-1, sequence_tensors_size).to(device)
                
                text_batch = text_tensors[batch_indices].view(-1, text_tensors_size).to(device)

                optimizer.zero_grad()
                decoded_text,  decoded_seq, _ = model(sequence_batch)
                loss = (criterion(decoded_text, text_batch) +
                        criterion(decoded_seq, sequence_batch))
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(data_loader)

            if phase == 'val' :#and epoch_loss < best_loss
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)
            elif phase == 'train':
                train_loss_history.append(epoch_loss)
        tqdm.tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss_history[-1]:.4f}, Val: {val_loss_history[-1]:.4f}")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, loss_vals


###############################################################################
# Data Preparation Functions
###############################################################################

def prepare_multimodal_data(seq_csv, text_csv, scale=True):
    """
    Load sequence and text representations (multi-column CSVs with an 'Entry' column), standardise each over its
    whole file (as for the dual autoencoder) and keep the proteins present in both.

    Returns:
        Tuple: (entries, sequence_tensors, text_tensors, seq_shift, seq_scale)
    """
    seq = rep_io.read_representation(seq_csv)
    text = rep_io.read_representation(text_csv, scale=scale)
    seq_shift = seq_scale = None
    if scale:
        seq_shift, seq_scale = rep_io.standardisation_factors(seq)
        seq = rep_io.apply_factors(seq, seq_shift, seq_scale)
    entries, (seq_v, text_v) = rep_io.align(seq, text)
    return (entries, torch.tensor(seq_v, dtype=torch.float), torch.tensor(text_v, dtype=torch.float),
            seq_shift, seq_scale)


def create_dataloaders(dataset_size, batch_size=128, validation_split=0.2, seed=42):
    """
    Create training and validation DataLoaders using indices.

    Args:
        dataset_size (int): Total number of data points.
        batch_size (int): Batch size for DataLoaders.
        validation_split (float): Fraction of the data to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple: (train_loader, validation_loader) containing DataLoader objects.
    """
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = DataLoader(train_indices, batch_size=batch_size, pin_memory=True, shuffle=True)
    validation_loader = DataLoader(val_indices, batch_size=batch_size, pin_memory=True, shuffle=True)
    return train_loader, validation_loader


###############################################################################
# Fused Representation Extraction
###############################################################################

def extract_fused_representation(model, sequence_tensors, entries, device):
    """
    Extract fused representation vectors using a trained sequence autoencoder.

    A forward hook is registered on the encoder4 layer of the model to capture the
    bottleneck (encoded) output. For each entry in the provided entries list, the corresponding
    sequence tensor is passed through the model, and the encoded representation is collected.

    Args:
        model (nn.Module): The trained Autoencoder_Seq model.
        sequence_tensors (Tensor): Tensor of sequence representations.
        entries (list): List of entry identifiers.
        device (torch.device): Device for model inference.

    Returns:
        pd.DataFrame: DataFrame with columns ['Entry', 'Vector'] where 'Vector' is the encoded representation,
                      converted to a multi-column format.
    """
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(entries), 512):
            _, _, encoded = model(sequence_tensors[start:start + 512].to(device))  # encoder4 output
            chunks.append(encoded.cpu().numpy())
    return rep_io.to_multi_col(entries, np.concatenate(chunks))


def save_fused_representation(fused_rep_df, output_csv_path):
    """
    Save the fused representation DataFrame to a CSV file.

    Args:
        fused_rep_df (pd.DataFrame): DataFrame with fused representations.
        output_csv_path (str): File path to save the CSV.
    """
    
    fused_rep_df.to_csv(output_csv_path, index=False)
    print(f"Fused representation saved at {output_csv_path}")
###############################################################################
# End of Module
###############################################################################
# Argument Parser
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TransferAE Training and Inference")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode: 'train' to train the model, 'test' to extract representations only.")

    # Common
    parser.add_argument("--seq_csv", type=str, required=True, help="Path to sequence CSV file")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--save_csv_path", type=str, required=True, help="Path to save output fused CSV")
    parser.add_argument("--representation_dim", type=int, default=None,
                        help="Ignored: the representation size is that of the dual autoencoder in --model_weights")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Only for training
   
    parser.add_argument("--text_csv", type=str, help="Path to text CSV file (required for train)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (only train)")
    parser.add_argument("--save_model_path", type=str, default="transfer_ae_weights.pth", help="Path to save trained model (only train)")
    parser.add_argument("--loss_plot_path", type=str, default="loss_curve.png",
                        help="Path to save the training/validation loss curve image.")
    parser.add_argument("--no_scaling", action="store_true",
                        help="Use the representations as given instead of standardising them (train)")
    # Only for test: per-dimension normalisation (x + shift) * scale; by default the factors saved with the
    # weights (<weights>.shift_factors.txt / .scaling_factors.txt) are used
    parser.add_argument("--shift_factors", type=str, default=None,
                        help="Text file with per-dimension shift factors (test mode, optional)")
    parser.add_argument("--scaling_factors", type=str, default=None,
                        help="Text file with per-dimension scaling factors (test mode, optional)")
    return parser.parse_args()
def plot_losses(train_hist, val_hist,save_path):
    plt.figure()
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(save_path)

# Seed Fix
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
def prepare_test_data(seq_csv):
    """Sequence representations (normalisation is applied by the caller) -> (float64 array, entries)."""
    seq = rep_io.read_representation(seq_csv)
    return seq.values.astype(np.float64), list(seq.index)


def ensure_parent_dirs(*paths):
    for path in paths:
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)


def check_dim(name, actual, expected, weights):
    if actual != expected:
        raise SystemExit("{} representations have {} dimensions but the model in {} expects {}.".format(
            name, actual, weights, expected))


# Main
if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device(os.environ.get("HOPER_DEVICE", "cpu"))  # set HOPER_DEVICE=cuda for GPU

    print(f"Using device: {device}")
    seq_csv = args.seq_csv
    ensure_parent_dirs(args.save_csv_path, args.save_model_path, args.loss_plot_path)

    if args.mode == "train":
        if not args.text_csv:
            raise SystemExit("Train mode requires --text_csv.")
        entries, sequence_tensors, text_tensors, seq_shift, seq_scale = prepare_multimodal_data(
            args.seq_csv, args.text_csv, scale=not args.no_scaling)

        # Dual (sequence + text) autoencoder with the sizes stored in its weights
        dual_state = torch.load(args.model_weights, map_location=device)
        dims = dual_dims_from_state(dual_state)
        check_dim("Sequence", sequence_tensors.shape[1], dims["seq_dim"], args.model_weights)
        check_dim("Text", text_tensors.shape[1], dims["text_dim"], args.model_weights)

    # Create training and validation DataLoaders
        train_loader, validation_loader = create_dataloaders(len(sequence_tensors), batch_size=args.batch_size,
                                                             seed=args.seed)

        model = Autoencoder(**dims).to(device)
        model.load_state_dict(dual_state)

    # Create sequence autoencoder model and load parameters from pre-trained model
        seq_model = Autoencoder_Seq(model).to(device)
        seq_model.load_parameters()
    
    # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(seq_model.parameters(), lr=0.001)
    
    # Train sequence autoencoder
        trained_model, train_loss_history, val_loss_history, loss_vals = train_model_for_seq(seq_model, train_loader, validation_loader, criterion, optimizer, args.epochs,sequence_tensors, text_tensors, device)
        plot_losses(train_loss_history, val_loss_history,args.loss_plot_path)
    # Save trained model
        torch.save(trained_model.state_dict(), args.save_model_path)
        shift_path, scale_path = rep_io.factor_paths(args.save_model_path)
        for path in (shift_path, scale_path):
            if os.path.exists(path):
                os.remove(path)
        if seq_shift is not None:
            rep_io.write_factors(shift_path, seq_shift)
            rep_io.write_factors(scale_path, seq_scale)
        print(f"Trained model saved to {args.save_model_path}")

        fused_rep_df = extract_fused_representation(trained_model, sequence_tensors, entries, device)
        fused_rep_df.to_csv(args.save_csv_path, index=False)
        print(f"Fused representations saved to {args.save_csv_path}")

    elif args.mode == "test":
        #from transfer_ae_components import prepare_test_data, Autoencoder_Seq, extract_fused_representation
        values, entries = prepare_test_data(seq_csv)
        # Same normalisation as in training: the factors saved with the weights, unless overridden.
        saved_shift, saved_scale = rep_io.factor_paths(args.model_weights)
        shift_file = args.shift_factors or (saved_shift if os.path.exists(saved_shift) else None)
        scale_file = args.scaling_factors or (saved_scale if os.path.exists(saved_scale) else None)
        if shift_file:
            values = values + rep_io.read_factors(shift_file)
        if scale_file:
            values = values * rep_io.read_factors(scale_file)
        print("Sequence normalisation: {}".format("(x + shift) * scale" if shift_file or scale_file else "none"))
        sequence_tensors = torch.tensor(values, dtype=torch.float)
        state = torch.load(args.model_weights, map_location=device)
        dims = dual_dims_from_state(state, prefix="pretrained_model.")
        check_dim("Sequence", sequence_tensors.shape[1], dims["seq_dim"], args.model_weights)
        model = Autoencoder_Seq(Autoencoder(**dims))
        model.load_state_dict(state)
        model = model.to(device)

        fused_rep_df = extract_fused_representation(model, sequence_tensors, entries, device)
        fused_rep_df.to_csv(args.save_csv_path, index=False)
        print(f"Fused representations saved to {args.save_csv_path}")
