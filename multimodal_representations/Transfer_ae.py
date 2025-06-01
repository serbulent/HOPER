import argparse
import os
import random
import copy
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler
class Autoencoder(nn.Module):
    """
    A multimodal autoencoder that fuses text, PPI, and sequence representations.
    The model contains three separate encoders for text, PPI, and sequence inputs,
    which are then concatenated and passed through a bottleneck layer (encoder4).
    The decoder reconstructs each modality from the bottleneck representation.
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
        self.ppi_dim = ppi_dim
        self.seq_dim = seq_dim
        self.zdim = zdim
        # Devam eden encoder ve decoder tanımlamaları...


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
        decoded_text = self.decoder_text(text_chunk)
        decoded_ppi = self.decoder_ppi(ppi_chunk)
        decoded_seq = self.decoder_seq(seq_chunk)
        return decoded_text, decoded_ppi, decoded_seq, z


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
        self.text_dim = 3072
        self.text_dim1 = 768
        self.text_dim2 = 512
        self.ppi_dim = 500
        self.ppi_dim1 = 1000
        self.ppi_dim2 = 1000
        self.seq_dim = 1024
        self.seq_dim1 = 768
        self.seq_dim2 = 512
        self.zdim = representation_dim

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
            nn.Linear(self.zdim, self.text_dim2 + self.ppi_dim2 + self.seq_dim2),
            nn.Tanh()
        )
        # Text decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(self.text_dim2, self.text_dim1),
            nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim),
            nn.Tanh()
        )
        # PPI decoder
        self.decoder2 = nn.Sequential(
            nn.Linear(self.ppi_dim2, self.ppi_dim1),
            nn.Tanh(),
            nn.Linear(self.ppi_dim1, self.ppi_dim),
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
    
    # PPI decoder (Autoencoder'da 'decoder_ppi' olarak tanımlı)
        self.decoder2[0].weight.data = copy.deepcopy(self.pretrained_model.decoder_ppi[0].weight.data)
        self.decoder2[2].weight.data = copy.deepcopy(self.pretrained_model.decoder_ppi[2].weight.data)
        self.decoder2[0].bias.data = copy.deepcopy(self.pretrained_model.decoder_ppi[0].bias.data)
        self.decoder2[2].bias.data = copy.deepcopy(self.pretrained_model.decoder_ppi[2].bias.data)
    
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
        decoded_ppi = self.decoder2(decoded_mid[:, self.text_dim2:self.text_dim2 + self.ppi_dim2])
        decoded_seq = self.decoder1(decoded_mid[:, self.text_dim2 + self.ppi_dim2:])
        return decoded_text, decoded_ppi, decoded_seq, encoded_mid


###############################################################################
# Training and Utility Functions
###############################################################################

def train_model_for_seq(model, train_loader, validation_loader, criterion, optimizer,
                        num_epochs, sequence_tensors, text_tensors, ppi_tensors, device):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    loss_vals = []

    # Get tensor sizes
    sequence_tensors_size = sequence_tensors.shape[1]
    ppi_tensors_size = ppi_tensors.shape[1]
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
                ppi_batch = ppi_tensors[batch_indices].view(-1, ppi_tensors_size).to(device)
                text_batch = text_tensors[batch_indices].view(-1, text_tensors_size).to(device)

                optimizer.zero_grad()
                decoded_text, decoded_ppi, decoded_seq, _ = model(sequence_batch)
                loss = (criterion(decoded_text, text_batch) +
                        criterion(decoded_ppi, ppi_batch) +
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
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, loss_vals


def convert_dataframe_to_multi_col(representation_dataframe):
    """
    Convert a DataFrame with vector representations into a multi-column format.

    Each element in the 'Vector' column (assumed to be a list/array) is split into
    its own column while retaining the identifier column ('Entry').

    Args:
        representation_dataframe (pd.DataFrame): DataFrame with columns 'Entry' and 'Vector'.

    Returns:
        pd.DataFrame: DataFrame with the 'Entry' column and one column per vector dimension.
    """
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry, right=vector, left_index=True, right_index=True)
    return multi_col_representation_vector


def convert_to_two_col(multi_col_representation_df):
    """
    Convert a multi-column representation DataFrame back to a two-column format.

    This function iterates through the multi-column DataFrame (excluding the 'Entry'
    column) and converts each row into a list of floats stored in a 'Vector' column.

    Args:
        multi_col_representation_df (pd.DataFrame): DataFrame where the first column is 'Entry'
                                                    and subsequent columns are vector dimensions.

    Returns:
        pd.DataFrame: DataFrame with two columns: 'Entry' and 'Vector' (list of floats).
    """
    vals = multi_col_representation_df.iloc[:, 1:]
    original_values_as_df = pd.DataFrame(columns=['Entry', 'Vector'])
    for index, row in tqdm.tqdm(vals.iterrows(), total=len(vals), desc="Converting to two-col"):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_df.iloc[index]['Entry'], list_of_floats]
    return original_values_as_df


###############################################################################
# Data Preparation Functions
###############################################################################

def prepare_multimodal_data(seq_csv, text_csv, ppi_csv):
    """
    Load and preprocess multimodal representation vectors from CSV files.

    The function reads sequence, text, and PPI representations (each CSV is expected
    to have an 'Entry' column and the remaining columns as the vector). Each modality
    is scaled using StandardScaler (excluding the 'Entry' column) and then converted into
    a two-column format (with columns 'Entry' and 'Vector'). Finally, the three modalities
    are merged on 'Entry'.

    Args:
        seq_csv (str): File path to the sequence representations CSV.
        text_csv (str): File path to the text representations CSV.
        ppi_csv (str): File path to the PPI representations CSV.

    Returns:
        pd.DataFrame: Merged DataFrame with columns ["Entry", "sequence", "ppi", "text"].
    """
    # Load CSV files
    seq_rep_multi_col = pd.read_csv(seq_csv)
    text_rep_multi_col = pd.read_csv(text_csv)
    ppi_rep_multi_col = pd.read_csv(ppi_csv)

    # Scale the vectors (exclude the 'Entry' column)
    scaler = StandardScaler()
    seq_rep_multi_col.loc[:, seq_rep_multi_col.columns != 'Entry'] = scaler.fit_transform(seq_rep_multi_col.loc[:, seq_rep_multi_col.columns != 'Entry'])
    text_rep_multi_col.loc[:, text_rep_multi_col.columns != 'Entry'] = scaler.fit_transform(text_rep_multi_col.loc[:, text_rep_multi_col.columns != 'Entry'])
    ppi_rep_multi_col.loc[:, ppi_rep_multi_col.columns != 'Entry'] = scaler.fit_transform(ppi_rep_multi_col.loc[:, ppi_rep_multi_col.columns != 'Entry'])

    # Convert to two-column format
    sequence_rep = convert_to_two_col(seq_rep_multi_col)
    ppi_rep = convert_to_two_col(ppi_rep_multi_col)
    text_rep = convert_to_two_col(text_rep_multi_col)

    # Merge the modalities
    fuse_phase_1 = sequence_rep.merge(ppi_rep, on='Entry')
    fuse_phase_2 = fuse_phase_1.merge(text_rep, on='Entry')
    fuse_phase_2.columns = ["Entry", "sequence", "ppi", "text"]

    return fuse_phase_2


def create_tensors_from_fused_df(fused_df):
    """
    Convert fused multimodal representations into PyTorch tensors.

    Args:
        fused_df (pd.DataFrame): DataFrame with columns ["Entry", "sequence", "ppi", "text"].
    
    Returns:
        Tuple: Three tensors for sequence, PPI, and text representations.
    """
    sequence_tensors = torch.tensor(list(fused_df['sequence'].values))
    ppi_tensors = torch.tensor(list(fused_df['ppi'].values))
    text_tensors = torch.tensor(list(fused_df['text'].values))
    return sequence_tensors, ppi_tensors, text_tensors


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
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register the forward hook on the encoder4 layer
    model.encoder4.register_forward_hook(get_activation('encoder4'))
    model.eval()

    fused_rep_ae = pd.DataFrame(columns=['Entry', 'Vector'])
    seq_tensor_size = sequence_tensors.shape[1]
    
    for i, entry in enumerate(tqdm.tqdm(entries, desc="Extracting Fused Representations")):
        seq_tensor = sequence_tensors[i].view(-1, seq_tensor_size).to(device)
        _ = model(seq_tensor)
        coding_layer_output = activation['encoder4'].tolist()[0]
        new_row = {'Entry': entry, 'Vector': coding_layer_output}
        fused_rep_ae = fused_rep_ae.append(new_row, ignore_index=True)
    
    fused_rep_ae_multi_col = convert_dataframe_to_multi_col(fused_rep_ae)
    return fused_rep_ae_multi_col


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
# Global
representation_dim = 512

# Argument Parser
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TransferAE Training and Inference")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode: 'train' to train the model, 'test' to extract representations only.")

    # Common
    parser.add_argument("--seq_csv", type=str, required=True, help="Path to sequence CSV file")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--save_csv_path", type=str, required=True, help="Path to save output fused CSV")
    parser.add_argument("--representation_dim", type=int, default=512, help="Dimension of representation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Only for training
    parser.add_argument("--ppi_csv", type=str, help="Path to PPI CSV file (required for train)")
    parser.add_argument("--text_csv", type=str, help="Path to text CSV file (required for train)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (only train)")
    parser.add_argument("--save_model_path", type=str, default="transfer_ae_weights.pth", help="Path to save trained model (only train)")

    return parser.parse_args()
def plot_losses(train_hist, val_hist):
    plt.figure()
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('/media/DATA2/sinem/isik_makale_1003/low_elenmis_yeni_datasetler/CC/CC_transfer/CC_loss_curve_Transfer_ae.png')

# Seed Fix
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
def prepare_test_data(seq_csv):
    """
    Convert fused multimodal representations into PyTorch tensors.

    Args:
        fused_df (pd.DataFrame): DataFrame with columns ["Entry", "sequence", "ppi", "text"].
    
    Returns:
        Tuple: Three tensors for sequence, PPI, and text representations.
    """
    seq_rep_multi_col = pd.read_csv(seq_csv)
   
    # Scale the vectors (exclude the 'Entry' column)
    scaler = StandardScaler()
    seq_rep_multi_col.loc[:, seq_rep_multi_col.columns != 'Entry'] = scaler.fit_transform(seq_rep_multi_col.loc[:, seq_rep_multi_col.columns != 'Entry'])
    sequence_rep = convert_to_two_col(seq_rep_multi_col)
    sequence_rep.columns = ["Entry", "sequence"]
    entries = sequence_rep["Entry"].tolist()
    sequence_tensors = torch.tensor(list(sequence_rep['sequence'].values))
    
    return sequence_tensors,entries

# Main
if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    print(f"Using device: {device}")
    epochs = args.epochs
    representation_dim = args.representation_dim
    seq_csv = args.seq_csv
    
    model_weights_path = args.model_weights
    batch_size=args.batch_size
    # Set device

    
    
    
    if args.mode == "train":
        # Prepare fused multimodal data
        fused_df = prepare_multimodal_data(args.seq_csv, args.text_csv, args.ppi_csv)
    
    # Create tensors from the fused DataFrame
        sequence_tensors, ppi_tensors, text_tensors = create_tensors_from_fused_df(fused_df)
        entries = fused_df["Entry"].tolist()
    
    # Create training and validation DataLoaders
        train_loader, validation_loader = create_dataloaders(len(sequence_tensors), batch_size=128)
        ppi_csv = args.ppi_csv
        text_csv = args.text_csv
        assert args.ppi_csv and args.text_csv, "Train mode requires ppi_csv and text_csv."
        
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
    
    # Create sequence autoencoder model and load parameters from pre-trained model
        seq_model = Autoencoder_Seq(model).to(device)
        seq_model.load_parameters()
    
    # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(seq_model.parameters(), lr=0.001)
    
    # Train sequence autoencoder
        trained_model, train_loss_history, val_loss_history, loss_vals = train_model_for_seq(seq_model, train_loader, validation_loader, criterion, optimizer, args.epochs,sequence_tensors, text_tensors, ppi_tensors, device)
        plot_losses(train_loss_history, val_loss_history)
    # Save trained model
        torch.save(trained_model.state_dict(), args.save_model_path)
        print(f"Trained model saved to {args.save_model_path}")

        fused_rep_df = extract_fused_representation(trained_model, sequence_tensors, entries, device)
        fused_rep_df.to_csv(args.save_csv_path, index=False)
        print(f"Fused representations saved to {args.save_csv_path}")

    elif args.mode == "test":
        #from transfer_ae_components import prepare_test_data, Autoencoder_Seq, extract_fused_representation
        
        sequence_tensors,entries = prepare_test_data(seq_csv)
        
        model = Autoencoder_Seq(Autoencoder())
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
        model = model.to(device)

        fused_rep_df = extract_fused_representation(model, sequence_tensors, entries, device)
        fused_rep_df.to_csv(args.save_csv_path, index=False)
        print(f"Fused representations saved to {args.save_csv_path}")
