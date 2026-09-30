# Example usage:
"""
# Train the autoencoder
python simple_ae.py train \\
  --fused_rep_path data/fused_train.csv \\
  --model_save_path models/autoencoder_best.pth \\
  --scaler_save_path models/scaler.pkl \\
  --output_csv outputs/simple_ae.csv \\
  --epochs 400 \\
  --batch_size 128 \\
  --learning_rate 0.001 \\
  --validation_split 0.2 \\
  --seed 42 \\
  --loss_plot_path outputs/loss_curve.png

# Run inference with pretrained model
python simple_ae.py inference \\
  --fused_rep_path data/fused_test.csv \\
  --model_load_path models/autoencoder_best.pth \\
  --scaler_load_path models/scaler.pkl \\
  --output_csv outputs/simple_ae_inference.csv
"""
#----------------------------
# 1. Autoencoder Definition
# ----------------------------
class Autoencoder(nn.Module):
    def __init__(self, representation_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = representation_dim  # For example, 2098
        self.layer1_dim = 1536
        self.layer2_dim = 1024
        self.layer3_dim = 768
        self.layer4_dim = 512

        activation_function = nn.Tanh()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.layer1_dim),
            activation_function,
            nn.Linear(self.layer1_dim, self.layer2_dim),
            activation_function,
            nn.Linear(self.layer2_dim, self.layer3_dim),
            activation_function,
        )

        self.encoder_mid = nn.Sequential(
            nn.Linear(self.layer3_dim, self.layer4_dim),
            activation_function
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.layer4_dim, self.layer3_dim),
            activation_function,
            nn.Linear(self.layer3_dim, self.layer2_dim),
            activation_function,
            nn.Linear(self.layer2_dim, self.layer1_dim),
            activation_function,
            nn.Linear(self.layer1_dim, self.input_dim),
            activation_function
        )

    def forward(self, fused_rep):
        encoded = self.encoder(fused_rep)
        encoded_mid = self.encoder_mid(encoded)
        decoded = self.decoder(encoded_mid)
        return decoded, encoded_mid

def plot_losses(train_hist, val_hist,save_path):
    plt.figure()
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(save_path)

# ----------------------------
# 2. Helper Functions
# ----------------------------
def convert_dataframe_to_multi_col(representation_dataframe):
    """
    Converts a single-column 'Vector' (list of floats) into a row-based multi-column DataFrame.
    Aligns 'Entry' column next to vector elements.
    """
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(
        left=entry.reset_index(drop=True),
        right=vector.reset_index(drop=True),
        left_index=True,
        right_index=True
    )
    return multi_col_representation_vector

def convert_to_two_col(multi_col_representation_df):
    """
    Converts a multi-column DataFrame (Entry + separate columns)
    into a two-column format: ['Entry', 'Vector'], where Vector is a list of floats.
    """
    vals = multi_col_representation_df.iloc[:, 1:]
    original_values_as_df = pd.DataFrame(columns=['Entry', 'Vector'])

    for index, row in tqdm.tqdm(vals.iterrows(), total=len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [
            multi_col_representation_df.iloc[index]['Entry'],
            list_of_floats
        ]

    return original_values_as_df

def get_activation(name, activation_dict):
    """
    Hook function to capture the output of encoder_mid layer during forward pass.
    """
    def hook(model, input, output):
        activation_dict[name] = output.detach()
    return hook

# ----------------------------
# 3. Training Function
# ----------------------------
def train_model(model, train_loader, validation_loader, criterion,
                optimizer, num_epochs, fused_tensors, device):
    """
    Trains the model and returns the weights corresponding to the best validation loss.
    Also returns the training and validation loss histories.
    """
    since = time.time()
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           verbose=True)

    fused_tensors_size = fused_tensors.shape[1]

    for epoch in tqdm.tqdm(range(num_epochs), desc="Epoch"):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = validation_loader

            running_loss = 0.0

            for batch_indices in loader:
                fused_batch = fused_tensors[batch_indices].view(-1, fused_tensors_size).to(device)

                optimizer.zero_grad()
                decoded_rep, _ = model(fused_batch)

                loss = criterion(decoded_rep, fused_batch)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(loader)

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                train_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.6f}')

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history

# ----------------------------
# 4. Training and Saving Model
# ----------------------------
def run_training(args):
    """
    Executes the training process:
    - Reads the CSV at fused_rep_path
    - Scales the data using StandardScaler
    - Splits data into train/val
    - Trains the model
    - Saves best weights and the scaler object
    - Generates simple_ae vectors and saves to CSV
    """
    fused_rep = pd.read_csv(args.fused_rep_path,index_col=0)
    if 'Entry' not in fused_rep.columns:
        raise ValueError("'Entry' column not found. Make sure your CSV contains a column named 'Entry'.")

    scaler = StandardScaler()
    cols_to_scale = [col for col in fused_rep.columns if col != 'Entry']
    fused_rep.loc[:, cols_to_scale] = scaler.fit_transform(fused_rep.loc[:, cols_to_scale])

    scaler_path = args.scaler_save_path if args.scaler_save_path else "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Scaler object saved to '{scaler_path}'.")

    fused_rep_two_col = convert_to_two_col(fused_rep)
    fused_tensors = torch.tensor(list(fused_rep_two_col['Vector'].values), dtype=torch.float32)

    batch_size = args.batch_size
    validation_split = args.validation_split
    shuffle_dataset = True
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    dataset_size = len(fused_rep_two_col)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_loader = DataLoader(train_indices, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(val_indices, batch_size=batch_size, shuffle=True, pin_memory=True)

    representation_dim = len(fused_rep_two_col['Vector'][0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = Autoencoder(representation_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    best_model, train_loss_hist, val_loss_hist = train_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        args.epochs,
        fused_tensors,
        device
    )
    plot_losses(train_loss_hist, val_loss_hist,args.loss_plot_path)

    model_save_path = args.model_save_path if args.model_save_path else "autoencoder_best.pth"
    torch.save(best_model.state_dict(), model_save_path)
    print(f"[INFO] Best model weights saved to '{model_save_path}'.")

    activation = {}
    best_model.encoder_mid.register_forward_hook(get_activation('encoder_mid', activation))

    simple_ae_rep = pd.DataFrame(columns=['Entry', 'Vector'])
    best_model.eval()
    fused_tensors_size = fused_tensors.shape[1]

    with torch.no_grad():
        for index, row in tqdm.tqdm(fused_rep_two_col.iterrows(), total=len(fused_rep_two_col),
                                   desc="Generating simple AE vectors"):
            fused_tensor = fused_tensors[index].view(1, fused_tensors_size).to(device)
            _ = best_model(fused_tensor)
            coding_layer_output = activation['encoder_mid'].tolist()[0]
            simple_ae_rep.loc[index] = [row['Entry'], coding_layer_output]

    simple_ae_multi_col = convert_dataframe_to_multi_col(simple_ae_rep)
    output_csv = args.output_csv if args.output_csv else "simple_ae.csv"
    simple_ae_multi_col.to_csv(output_csv, index=False)
    print(f"[INFO] Simple AE multi-column output saved to '{output_csv}'.")

# ----------------------------
# 5. Inference Function
# ----------------------------
def run_inference(args):
    """
    Executes inference using a pretrained autoencoder:
    - Loads model weights and scaler
    - Reads input data and applies the same scaling
    - Extracts encoder_mid vectors
    - Saves output as multi-column CSV
    """
    if not os.path.exists(args.model_load_path):
        raise FileNotFoundError(f"[ERROR] Model weights not found: {args.model_load_path}")
    if not os.path.exists(args.scaler_load_path):
        raise FileNotFoundError(f"[ERROR] Scaler object not found: {args.scaler_load_path}")

    with open(args.scaler_load_path, 'rb') as f:
        scaler: StandardScaler = pickle.load(f)
    print(f"[INFO] Scaler loaded from '{args.scaler_load_path}'.")

    fused_rep = pd.read_csv(args.fused_rep_path,index_col=0)
    if 'Entry' not in fused_rep.columns:
        raise ValueError("'Entry' column not found. Make sure your CSV contains a column named 'Entry'.")

    cols_to_scale = [col for col in fused_rep.columns if col != 'Entry']
    fused_rep.loc[:, cols_to_scale] = scaler.transform(fused_rep.loc[:, cols_to_scale])

    fused_rep_two_col = convert_to_two_col(fused_rep)
    fused_tensors = torch.tensor(list(fused_rep_two_col['Vector'].values), dtype=torch.float32)

    representation_dim = len(fused_rep_two_col['Vector'][0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = Autoencoder(representation_dim).to(device)
    model.load_state_dict(torch.load(args.model_load_path, map_location=device))
    model.eval()
    print(f"[INFO] Model weights loaded from '{args.model_load_path}'.")

    activation = {}
    model.encoder_mid.register_forward_hook(get_activation('encoder_mid', activation))

    simple_ae_rep = pd.DataFrame(columns=['Entry', 'Vector'])
    fused_tensors_size = fused_tensors.shape[1]

    with torch.no_grad():
        for index, row in tqdm.tqdm(fused_rep_two_col.iterrows(), total=len(fused_rep_two_col),
                                   desc="Inference - Generating AE vectors"):
            fused_tensor = fused_tensors[index].view(1, fused_tensors_size).to(device)
            _ = model(fused_tensor)
            coding_layer_output = activation['encoder_mid'].tolist()[0]
            simple_ae_rep.loc[index] = [row['Entry'], coding_layer_output]

    simple_ae_multi_col = convert_dataframe_to_multi_col(simple_ae_rep)
    output_csv = args.output_csv if args.output_csv else "simple_ae_inference.csv"
    simple_ae_multi_col.to_csv(output_csv, index=False)
    print(f"[INFO] Inference output saved to '{output_csv}'.")

# ----------------------------
# 6. Main Function and Argparse
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Script for training and/or inference using an Autoencoder."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Choose mode: 'train' or 'inference'.")

    # 6.1. Train mode arguments
    train_parser = subparsers.add_parser("train", help="Train the model and generate simple AE outputs.")
    train_parser.add_argument("--fused_rep_path", type=str, required=True,
                              help="Path to the CSV file containing fused representation data.")
    train_parser.add_argument("--model_save_path", type=str, required=False, default="autoencoder_best.pth",
                              help="Path to save the best model weights after training.")
    train_parser.add_argument("--scaler_save_path", type=str, required=False, default="scaler.pkl",
                              help="Path to save the scaler object during training.")
    train_parser.add_argument("--output_csv", type=str, required=False, default="simple_ae.csv",
                              help="Path to save the multi-column simple AE output CSV.")
    train_parser.add_argument("--epochs", type=int, required=False, default=400,
                              help="Number of training epochs.")
    train_parser.add_argument("--batch_size", type=int, required=False, default=128,
                              help="Mini-batch size.")
    train_parser.add_argument("--learning_rate", type=float, required=False, default=1e-3,
                              help="Learning rate for optimizer.")
    train_parser.add_argument("--validation_split", type=float, required=False, default=0.2,
                              help="Validation set ratio (between 0 and 1).")
    train_parser.add_argument("--seed", type=int, required=False, default=42,
                              help="Random seed for reproducibility.")
    train_parser.add_argument("--loss_plot_path", type=str, default="loss_curve.png",
                        help="Path to save the training/validation loss curve image.")

    # 6.2. Inference mode arguments
    infer_parser = subparsers.add_parser("inference", help="Run inference using a trained model.")
    infer_parser.add_argument("--fused_rep_path", type=str, required=True,
                              help="Path to the CSV file containing fused representation data for inference.")
    infer_parser.add_argument("--model_load_path", type=str, required=True,
                              help="Path to the pre-trained model weights (.pth file).")
    infer_parser.add_argument("--scaler_load_path", type=str, required=True,
                              help="Path to the previously saved scaler object (.pkl file).")
    infer_parser.add_argument("--output_csv", type=str, required=False, default="simple_ae_inference.csv",
                              help="Path to save the multi-column inference output CSV.")

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "inference":
        run_inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
