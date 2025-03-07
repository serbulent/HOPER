
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
#from ranking import *
from sklearn.preprocessing import StandardScaler
import tqdm

import time
import copy
import matplotlib.pyplot as plt
import itertools
import random
import os

representation_dim = 512
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.text_dim = 1024
        self.text_dim1 = 768
        self.text_dim2 = 512
        self.ppi_dim = 10
        self.ppi_dim1 = 10
        self.ppi_dim2 = 10
        self.seq_dim = 1024
        self.seq_dim1 = 768
        self.seq_dim2 = 512
        self.zdim = representation_dim

        self.encoder1 = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim1),
            nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim2),
            nn.Tanh()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(self.ppi_dim, self.ppi_dim1),
            nn.Tanh(),
            nn.Linear(self.ppi_dim1, self.ppi_dim2),
            nn.Tanh()
        )
        self.encoder3 = nn.Sequential(
            nn.Linear(self.seq_dim, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim2),
            nn.Tanh()
        )
        self.encoder4 = nn.Sequential(
            nn.Linear(self.text_dim2 + self.ppi_dim2 + self.seq_dim2, self.zdim),
            nn.Tanh()
        )

        self.decoder4 = nn.Sequential(
            nn.Linear(self.zdim, self.text_dim2 + self.ppi_dim2 + self.seq_dim2),
            nn.Tanh()
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(self.text_dim2, self.text_dim1),
            nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim),
            nn.Tanh()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.ppi_dim2, self.ppi_dim1),
            nn.Tanh(),
            nn.Linear(self.ppi_dim1, self.ppi_dim),
            nn.Tanh()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.seq_dim2, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim),
            nn.Tanh()
        ) 
        
    def forward(self, x_text, x_ppi, x_seq):
        encoded_text = self.encoder1(x_text)
        encoded_ppi = self.encoder2(x_ppi)
        encoded_seq = self.encoder3(x_seq)
        encoded_mid = self.encoder4(torch.cat((encoded_text, encoded_ppi, encoded_seq), dim=1))
        decoded_mid = self.decoder4(encoded_mid)
        decoded_text = self.decoder3(decoded_mid[:, 0:self.text_dim2])
        decoded_ppi = self.decoder2(decoded_mid[:, self.text_dim2:self.text_dim2 + self.ppi_dim2])
        decoded_seq = self.decoder1(decoded_mid[:, self.text_dim2 + self.ppi_dim2:])
        return decoded_text, decoded_ppi, decoded_seq, encoded_mid
epochs = 500
# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create an optimizer object
# Adam optimizer with learning rate 1e-3


model = Autoencoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_model=model.load_state_dict(torch.load("/media/DATA2/sinem/isik_makale_1003/multimodel_weights.pth",map_location=torch.device('cpu')))


class Autoencoder_Seq(nn.Module):
    def __init__(self,pretrained_model):
        super(Autoencoder_Seq, self).__init__()
        self.pretrained_model = pretrained_model
        
        self.text_dim = 1024
        self.text_dim1 = 768
        self.text_dim2 = 512
        self.ppi_dim = 10
        self.ppi_dim1 = 10
        self.ppi_dim2 = 10
        self.seq_dim = 1024
        self.seq_dim1 = 768
        self.seq_dim2 = 512
        self.zdim = representation_dim

        self.encoder3 = nn.Sequential(
            nn.Linear(self.seq_dim, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim2),
            nn.Tanh()
        )
        self.encoder4 = nn.Sequential(
            nn.Linear(self.seq_dim2, self.zdim),
            nn.Tanh()
        )

        self.decoder4 = nn.Sequential(
            nn.Linear(self.zdim, self.text_dim2 + self.ppi_dim2 + self.seq_dim2),
            nn.Tanh()
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(self.text_dim2, self.text_dim1),
            nn.Tanh(),
            nn.Linear(self.text_dim1, self.text_dim),
            nn.Tanh()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.ppi_dim2, self.ppi_dim1),
            nn.Tanh(),
            nn.Linear(self.ppi_dim1, self.ppi_dim),
            nn.Tanh()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.seq_dim2, self.seq_dim1),
            nn.Tanh(),
            nn.Linear(self.seq_dim1, self.seq_dim),
            nn.Tanh()
        )
        
    def load_parameters(self):
        self.encoder3[0].weight.data = copy.deepcopy(self.pretrained_model.encoder3[0].weight.data)
        self.encoder3[2].weight.data = copy.deepcopy(self.pretrained_model.encoder3[2].weight.data)
        self.encoder3[0].bias.data = copy.deepcopy(self.pretrained_model.encoder3[0].bias.data)
        self.encoder3[2].bias.data = copy.deepcopy(self.pretrained_model.encoder3[2].bias.data)
        
        self.decoder4[0].weight.data = copy.deepcopy(self.pretrained_model.decoder4[0].weight.data)
        self.decoder4[0].bias.data = copy.deepcopy(self.pretrained_model.decoder4[0].bias.data)
        
        self.decoder3[0].weight.data = copy.deepcopy(self.pretrained_model.decoder3[0].weight.data)
        self.decoder3[2].weight.data = copy.deepcopy(self.pretrained_model.decoder3[2].weight.data)
        self.decoder3[0].bias.data = copy.deepcopy(self.pretrained_model.decoder3[0].bias.data)
        self.decoder3[2].bias.data = copy.deepcopy(self.pretrained_model.decoder3[2].bias.data)
        
        self.decoder2[0].weight.data = copy.deepcopy(self.pretrained_model.decoder2[0].weight.data)
        self.decoder2[2].weight.data = copy.deepcopy(self.pretrained_model.decoder2[2].weight.data)
        self.decoder2[0].bias.data = copy.deepcopy(self.pretrained_model.decoder2[0].bias.data)
        self.decoder2[2].bias.data = copy.deepcopy(self.pretrained_model.decoder2[2].bias.data)
        
        self.decoder1[0].weight.data = copy.deepcopy(self.pretrained_model.decoder1[0].weight.data)
        self.decoder1[2].weight.data = copy.deepcopy(self.pretrained_model.decoder1[2].weight.data)
        self.decoder1[0].bias.data = copy.deepcopy(self.pretrained_model.decoder1[0].bias.data)
        self.decoder1[2].bias.data = copy.deepcopy(self.pretrained_model.decoder1[2].bias.data)

    def forward(self, x_seq):
        encoded_seq = self.encoder3(x_seq)
        encoded_mid = self.encoder4(encoded_seq)
        decoded_mid = self.decoder4(encoded_mid)
        decoded_text = self.decoder3(decoded_mid[:, 0:self.text_dim2])
        decoded_ppi = self.decoder2(decoded_mid[:, self.text_dim2:self.text_dim2 + self.ppi_dim2])
        decoded_seq = self.decoder1(decoded_mid[:, self.text_dim2 + self.ppi_dim2:])
        return decoded_text, decoded_ppi, decoded_seq, encoded_mid

def train_model_for_seq(model, train_loader, validation_loader, criterion,optimizer, num_epochs,\
                sequence_tensors,text_tensors,ppi_tensors):
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10**10    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True)
    loss_vals = []
    encoded_mid = None
    sequence_tensors_size = sequence_tensors.shape[1]
    ppi_tensors_size = ppi_tensors.shape[1]
    text_tensors_size = text_tensors.shape[1]
    for epoch in tqdm.tqdm(range(num_epochs)):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                inputs = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                inputs = validation_loader

            loss = 0
            for batch_features in inputs:
                sequence_batch = sequence_tensors[batch_features].view(-1, sequence_tensors_size).to(device)
                ppi_batch = ppi_tensors[batch_features].view(-1, ppi_tensors_size).to(device)
                text_batch = text_tensors[batch_features].view(-1, text_tensors_size).to(device)             
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()                
                decoded_text, decoded_ppi, decoded_seq, encoded_mid =\
                model(sequence_batch)
                # compute training reconstruction loss
                train_loss = criterion(decoded_text, text_batch) + \
                criterion(decoded_ppi, ppi_batch) + criterion(decoded_seq, sequence_batch) 
                #train_loss = criterion(outputs, batch_features,target)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
            # compute the epoch training loss
            epoch_loss = loss / len(inputs)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)
            elif phase == 'train':
                train_loss_history.append(epoch_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history,val_loss_history,loss_vals

epochs = 500
# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
best_model = Autoencoder_Seq(model).to(device)

best_model.encoder3[0].weight.data
len(best_model.encoder3)


model = Autoencoder_Seq(best_model).to(device)
model.load_parameters()
# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()
#criterion = nn.CosineEmbeddingLoss()

def convert_dataframe_to_multi_col(representation_dataframe):
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry,right=vector,left_index=True, right_index=True)
    return multi_col_representation_vector

def convert_to_two_col(multi_col_representation_df):
    vals = multi_col_representation_df.iloc[:,1:(len(multi_col_representation_df.columns))]
    original_values_as_df = pd.DataFrame(columns=['Entry', 'Vector'])
    for index, row in tqdm.tqdm(vals.iterrows(), total = len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_df.iloc[index]['Entry']] + [list_of_floats]
    return original_values_as_df

scaler = StandardScaler()
#best_model.encoder3[0].weight.data
seq_rep_multi_col = pd.read_csv("/media/DATA/serbulent/DATA/Thesis/ReviewPaper/generalized_representation_benchmark/DATA/representation_vectors/revision-1/T5_UNIPROT_HUMAN.csv")

text_rep_multi_col =  pd.read_csv("/media/DATA/serbulent/DATA/Thesis/TheRepresentation/ProtText/representation_vectors/unipubmed_tfidf_vectors_pca1024_multi_col_vec.csv")

ppi_rep_multi_col = pd.read_csv("/media/DATA/serbulent/DATA/Thesis/TheRepresentation/PPI-Reps/Node2Vec/d_10_p_2_q_1.csv")


scaler = StandardScaler()
seq_rep_multi_col.loc[:, seq_rep_multi_col.columns != 'Entry'] = \
scaler.fit_transform(seq_rep_multi_col.loc[:, seq_rep_multi_col.columns != 'Entry'])

ppi_rep_multi_col.loc[:, ppi_rep_multi_col.columns != 'Entry'] = \
scaler.fit_transform(ppi_rep_multi_col.loc[:, ppi_rep_multi_col.columns != 'Entry'])

text_rep_multi_col.loc[:, text_rep_multi_col.columns != 'Entry'] = \
scaler.fit_transform(text_rep_multi_col.loc[:, text_rep_multi_col.columns != 'Entry'])
sequence_rep = convert_to_two_col(seq_rep_multi_col)
ppi_rep = convert_to_two_col(ppi_rep_multi_col)
text_rep = convert_to_two_col(text_rep_multi_col)
fuse_phase_1 = sequence_rep.merge(ppi_rep, on='Entry')
fuse_phase_2 = fuse_phase_1.merge(text_rep, on='Entry')
fuse_phase_2.columns = ["Entry","sequence", "ppi","text"]
sequence_tensors = torch.tensor(list(fuse_phase_2['sequence'].values))
ppi_tensors = torch.tensor(list(fuse_phase_2['ppi'].values))
text_tensors = torch.tensor(list(fuse_phase_2['text'].values))
batch_size = 128
validation_split = .2
shuffle_dataset = True
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# Creating data indices for training and validation splits:
dataset_size = len(sequence_tensors)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_loader = DataLoader(train_indices, batch_size=batch_size, 
                                            pin_memory=True,shuffle=True)
validation_loader = DataLoader(val_indices, batch_size=batch_size,
                                                 pin_memory=True,shuffle=True)


representation_dim = 512




best_model_seq, train_loss_history_seq,val_loss_history_seq,loss_vals_seq\
= train_model_for_seq(model, train_loader, validation_loader, criterion,\
              optimizer, epochs,sequence_tensors,text_tensors,ppi_tensors)
plt.title("Training and Validation Loss")
plt.xlabel("Training Epochs")
plt.ylabel("MSE Loss")
plt.plot(range(1,epochs+1),train_loss_history_seq,label="Training")
plt.plot(range(1,epochs+1),val_loss_history_seq,label="Validation")
plt.ylim((0.75,2.5))
plt.xticks(np.arange(1, epochs, 100.0))
plt.legend()
plt.rcParams["figure.figsize"] = (10,10)
plt.show()
# create and register an forward hook
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

best_model_seq.encoder4.register_forward_hook(get_activation('encoder4'))
#x = fused_rep_torch_tensor[0].to(device)
#best_model.eval()
#with torch.no_grad():
    #output = best_model(x)
#activation['encoded_mid'].shape
sequence_tensors_size = sequence_tensors.shape[1]
ppi_tensors_size = ppi_tensors.shape[1]
text_tensors_size = text_tensors.shape[1]

best_model_seq.eval()
with torch.no_grad():
    output = best_model_seq(sequence_tensors[0].view(-1, sequence_tensors_size).to(device))

fused_rep_ae = pd.DataFrame(columns=['Entry', 'Vector'])
#sequence_tensors_size = sequence_tensors.shape[1]
#ppi_tensors_size = ppi_tensors.shape[1]
#text_tensors_size = text_tensors.shape[1]
best_model.eval()
with torch.no_grad():
    for index,row in tqdm.tqdm(fuse_phase_2.iterrows(),total=len(fuse_phase_2)):
        #fused_rep_tensor = torch.tensor(list(row['Vector']))
        text_tensor = text_tensors[index].view(-1, text_tensors_size).to(device)
        ppi_tensor = ppi_tensors[index].view(-1, ppi_tensors_size).to(device)
        seq_tensor = sequence_tensors[index].view(-1, sequence_tensors_size).to(device)
        
        _ = best_model_seq(seq_tensor)
        coding_layer_output = activation['encoder4'].tolist()[0]    
        new_row = {'Entry':row['Entry'], 'Vector':coding_layer_output}
        fused_rep_ae = fused_rep_ae.append(new_row, ignore_index=True)
multi_modal_rep_ae_multi_col = convert_dataframe_to_multi_col(fused_rep_ae)

multi_modal_rep_ae_multi_col.to_csv\
("/media/DATA2/sinem/isik_makale_1003/"+str(representation_dim)+"_dim_"+str(epochs)+"_epochs_ppi_no_comp.csv"\
,index=False)