import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import *
import numpy as np
import pickle
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tqdm
import time
import copy
import os
import random
import matplotlib.pyplot as plt
# ----------------------------
# 1. Autoencoder Tanımı
# ----------------------------
class Autoencoder(nn.Module):
    def __init__(self, representation_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = representation_dim  # Örneğin 2098
        self.layer1_dim = 1536
        self.layer2_dim = 1024
        self.layer3_dim = 768
        self.layer4_dim = 512

        activation_function = nn.Tanh()

        # Encoder katmanları
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

        # Decoder katmanları
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

def plot_losses(train_hist, val_hist):
    plt.figure()
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig("/media/DATA2/testuser2/deneme_aybar_hoca/HOPER/multimodal_representations/cc_simple_ae/CC_simple_ae_representations_loss_curve_.png")

# ----------------------------
# 2. Yardımcı Fonksiyonlar
# ----------------------------
def convert_dataframe_to_multi_col(representation_dataframe):
    """
    Tek sütunlu 'Vector' dizisini (list of floats) satır bazlı multi-column DataFrame'e çevirir.
    'Entry' sütunu ve vektör elemanlarını yan yana getirir.
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
    Multi-column DataFrame'i (Entry + ayrı ayrı sütunlar) alır,
    her satırdaki vektör elemanlarını tek bir liste haline getirip
    DataFrame'i iki sütunlu hale getirir: ['Entry', 'Vector'].
    """
    vals = multi_col_representation_df.iloc[:, 1:]
    original_values_as_df = pd.DataFrame(columns=['Entry', 'Vector'])

    for index, row in tqdm.tqdm(vals.iterrows(), total=len(vals)):
        # Her satırdaki vektör elemanlarını float listesine çevir
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [
            multi_col_representation_df.iloc[index]['Entry'],
            list_of_floats
        ]

    return original_values_as_df


def get_activation(name, activation_dict):
    """
    Modelin encoder_mid katman çıkışını hook ile yakalayabilmek için.
    """
    def hook(model, input, output):
        activation_dict[name] = output.detach()
    return hook


# ----------------------------
# 3. Eğitim Fonksiyonu
# ----------------------------
def train_model(model, train_loader, validation_loader, criterion,
                optimizer, num_epochs, fused_tensors, device):
    """
    Modeli eğitir ve en iyi doğrulama kaybı (val_loss) elde edilen ağırlıkları döner.
    Ayrıca eğitim ve doğrulama kayıp geçmişini döner.
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
                # batch_indices, fused_tensors'teki satırların indekslerini içeriyor
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

    # En iyi ağırlıkları yükle
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history


# ----------------------------
# 4. Eğitim ve Model Kaydetme
# ----------------------------
def run_training(args):
    """
    Eğitim işlemini gerçekleştirir:
    - fused_rep_path'teki CSV'yi okur
    - StandardScaler ile ölçekler
    - Dataları train/val olarak ayırır
    - Modeli eğitir
    - En iyi ağırlıkları ve scaler objesini kaydeder
    - simple_ae vektörlerini oluşturup CSV'e kaydeder
    """
    # 4.1. Veri Okuma ve Ölçekleme
    fused_rep = pd.read_csv(args.fused_rep_path,index_col=0)
    if 'Entry' not in fused_rep.columns:
        raise ValueError("'Entry' sütunu bulunamadı. Lütfen CSV'de 'Entry' adında bir sütun olduğundan emin olun.")

    scaler = StandardScaler()
    # "Entry" dışındaki tüm sütunları ölçeklendir
    cols_to_scale = [col for col in fused_rep.columns if col != 'Entry']
    fused_rep.loc[:, cols_to_scale] = scaler.fit_transform(fused_rep.loc[:, cols_to_scale])

    # Scaler'ı disk'e kaydet
    scaler_path = args.scaler_save_path if args.scaler_save_path else "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Scaler objesi '{scaler_path}' olarak kaydedildi.")
    #breakpoint()
    # 4.2. Two-col formatına çevir
    fused_rep_two_col = convert_to_two_col(fused_rep)
    fused_tensors = torch.tensor(list(fused_rep_two_col['Vector'].values), dtype=torch.float32)

    # 4.3. DataLoader Hazırlığı
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

    # 4.4. Model Oluşturma ve Eğitme
    representation_dim = len(fused_rep_two_col['Vector'][0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Kullanılan cihaz: {device}")

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
    plot_losses(train_loss_hist, val_loss_hist)
    # 4.5. Model Ağırlıklarını Kaydet
    model_save_path = args.model_save_path if args.model_save_path else "autoencoder_best.pth"
    torch.save(best_model.state_dict(), model_save_path)
    print(f"[INFO] En iyi model ağırlıkları '{model_save_path}' olarak kaydedildi.")

    # 4.6. Simple AE Vektörleri Oluşturma (Training Data Üzerinden)
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

    # 4.7. Multi-column formatına çevir ve kaydet
    simple_ae_multi_col = convert_dataframe_to_multi_col(simple_ae_rep)
    output_csv = args.output_csv if args.output_csv else "simple_ae.csv"
    simple_ae_multi_col.to_csv(output_csv, index=False)
    print(f"[INFO] Simple AE multi-column çıktısı '{output_csv}' olarak kaydedildi.")


# ----------------------------
# 5. İnference Fonksiyonu
# ----------------------------
def run_inference(args):
    """
    İnference işlemini gerçekleştirir:
    - Daha önce eğitilmiş model ağırlıklarını ve scaler objesini yükler
    - fused_rep_path'teki veriyi okur ve aynı scaler ile ölçekler
    - Modeli yükler, encoder_mid katmanına hook ekler
    - Her satır için encoding (encoded_mid) vektörünü hesaplar
    - Sonucu multi-column formatında CSV olarak kaydeder
    """
    # 5.1. Ölçekleyici ve Model Yükleme
    if not os.path.exists(args.model_load_path):
        raise FileNotFoundError(f"[ERROR] Model ağırlıkları bulunamadı: {args.model_load_path}")
    if not os.path.exists(args.scaler_load_path):
        raise FileNotFoundError(f"[ERROR] Scaler objesi bulunamadı: {args.scaler_load_path}")

    with open(args.scaler_load_path, 'rb') as f:
        scaler: StandardScaler = pickle.load(f)
    print(f"[INFO] Scaler objesi '{args.scaler_load_path}' yüklendi.")

    # 5.2. Veriyi Okuma ve Ölçekleme
    fused_rep = pd.read_csv(args.fused_rep_path,index_col=0)
    if 'Entry' not in fused_rep.columns:
        raise ValueError("'Entry' sütunu bulunamadı. Lütfen CSV'de 'Entry' adında bir sütun olduğundan emin olun.")

    cols_to_scale = [col for col in fused_rep.columns if col != 'Entry']
    fused_rep.loc[:, cols_to_scale] = scaler.transform(fused_rep.loc[:, cols_to_scale])

    # 5.3. Two-col formatına çevir
    fused_rep_two_col = convert_to_two_col(fused_rep)
    fused_tensors = torch.tensor(list(fused_rep_two_col['Vector'].values), dtype=torch.float32)

    # 5.4. Model Tanımlama ve Yükleme
    representation_dim = len(fused_rep_two_col['Vector'][0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Kullanılan cihaz: {device}")

    model = Autoencoder(representation_dim).to(device)
    model.load_state_dict(torch.load(args.model_load_path, map_location=device))
    model.eval()
    print(f"[INFO] Model ağırlıkları '{args.model_load_path}' yüklendi.")

    activation = {}
    model.encoder_mid.register_forward_hook(get_activation('encoder_mid', activation))

    # 5.5. Encoding (Inference) ve Çıktı Üretimi
    simple_ae_rep = pd.DataFrame(columns=['Entry', 'Vector'])
    fused_tensors_size = fused_tensors.shape[1]

    with torch.no_grad():
        for index, row in tqdm.tqdm(fused_rep_two_col.iterrows(), total=len(fused_rep_two_col),
                                   desc="Inference - Generating AE vectors"):
            fused_tensor = fused_tensors[index].view(1, fused_tensors_size).to(device)
            _ = model(fused_tensor)
            coding_layer_output = activation['encoder_mid'].tolist()[0]
            simple_ae_rep.loc[index] = [row['Entry'], coding_layer_output]

    # 5.6. Multi-column formatına çevir ve kaydet
    simple_ae_multi_col = convert_dataframe_to_multi_col(simple_ae_rep)
    output_csv = args.output_csv if args.output_csv else "simple_ae_inference.csv"
    simple_ae_multi_col.to_csv(output_csv, index=False)
    print(f"[INFO] Inference sonucu multi-column çıktısı '{output_csv}' olarak kaydedildi.")


# ----------------------------
# 6. Ana Fonksiyon ve Argparse
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Autoencoder ile eğitim ve/veya inference işlemleri için script."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Mode 'train' veya 'inference' seçin.")

    # 6.1. Train modu argümanları
    train_parser = subparsers.add_parser("train", help="Modeli eğit ve simple AE çıktılarını oluştur.")
    train_parser.add_argument("--fused_rep_path", type=str, required=True,
                              help="Fusion edilmiş temsil verilerini içeren CSV dosyasının yolu.")
    train_parser.add_argument("--model_save_path", type=str, required=False, default="autoencoder_best.pth",
                              help="Eğitim sonrası en iyi model ağırlıklarının kaydedileceği dosya.")
    train_parser.add_argument("--scaler_save_path", type=str, required=False, default="scaler.pkl",
                              help="Eğitim sırasında scaler objesinin kaydedileceği dosya.")
    train_parser.add_argument("--output_csv", type=str, required=False, default="simple_ae.csv",
                              help="Eğitim sonunda oluşturulan multi-column simple AE çıktısının kaydedileceği CSV.")
    train_parser.add_argument("--epochs", type=int, required=False, default=400,
                              help="Eğitim epoch sayısı.")
    train_parser.add_argument("--batch_size", type=int, required=False, default=128,
                              help="Mini-batch boyutu.")
    train_parser.add_argument("--learning_rate", type=float, required=False, default=1e-3,
                              help="Optimizer için öğrenme hızı (lr).")
    train_parser.add_argument("--validation_split", type=float, required=False, default=0.2,
                              help="Validation set oranı (0-1 arasında).")
    train_parser.add_argument("--seed", type=int, required=False, default=42,
                              help="Rastgelelik için seed değeri.")

    # 6.2. Inference modu argümanları
    infer_parser = subparsers.add_parser("inference", help="Eğitilmiş modeli kullanarak inference yap.")
    infer_parser.add_argument("--fused_rep_path", type=str, required=True,
                              help="Inference için fusion edilmiş temsil verilerini içeren CSV dosyasının yolu.")
    infer_parser.add_argument("--model_load_path", type=str, required=True,
                              help="Önceden eğitilmiş model ağırlıklarının bulunduğu .pth dosyasının yolu.")
    infer_parser.add_argument("--scaler_load_path", type=str, required=True,
                              help="Önceden kaydedilmiş scaler objesinin bulunduğu .pkl dosyasının yolu.")
    infer_parser.add_argument("--output_csv", type=str, required=False, default="simple_ae_inference.csv",
                              help="Inference sonucu multi-column simple AE çıktısının kaydedileceği CSV.")

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "inference":
        run_inference(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
