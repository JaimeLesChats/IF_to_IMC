import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader
import tqdm

# NEURAL NETWORK

class SimpleNN(nn.Module):
    def __init__(self, nb_in, nb_out, hidden_dim=1000):
        super().__init__()

        self.in_to_hidden1 = nn.Linear(nb_in,hidden_dim)
        self.hidden1_to_hidden2 = nn.Linear(hidden_dim,hidden_dim)
        self.hidden2_to_out = nn.Linear(hidden_dim,nb_out)

        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.in_to_hidden1,self.relu,
                                 self.hidden1_to_hidden2,self.relu,
                                 self.hidden2_to_out)

    def forward(self,input):
        prediction = self.net(input)
        
        return prediction

#Dataset

X_test = torch.linspace(-10, 10, 100).view(-1, 1)
Y_test = X_test * 2

class MarkersDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor(X_test)
        self.y = torch.tensor(Y_test)
        #self.X = torch.tensor(markers_df.to_numpy())
        #self.y = torch.tensor(clusters_df.to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# FUNCTIONS

def create_checkpoint(model, optimizer, epochs):
     checkpoint = {'epochs' : epochs,
                   'model_state_dict' : model.state_dict(),
                   'optimizer_state_dict' : optimizer.state_dict(),
                   'loss' : None}
     
     return checkpoint

def save_nn(checkpoint):
    directory = './scripts/NN/nn_checkpoints/'

    if not os.path.exists(directory):
            os.makedirs(directory)

    nn_basename = str(checkpoint['epochs']) + '_epochs.pth'
    nn_path = directory + nn_basename

    torch.save(checkpoint,nn_path)

def load_nn(model, optimizer, nn_path):
    checkpoint = torch.load(nn_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epochs']

def train(X_train,Y_train,model, optimizer, loss_f, epochs_done, epochs = 50,):

    loop = tqdm.tqdm(range(epochs_done+1,epochs_done+epochs+1),
                     ncols=100, colour='green', desc='Training ... ', )

    for epoch in loop:
        model.train()
        predicted_Y= model(X_train)

        reconstruction_loss = loss_f(predicted_Y,Y_train)
        loss = reconstruction_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
    
    checkpoint = create_checkpoint(model,optimizer,epochs_done+epochs)
    return checkpoint

def most_trained_path(directory):
    max_epoch = 0
    for filename in os.listdir(directory):
        epochs = int(re.split(r'_',filename)[0])
        max_epoch = max(max_epoch,epochs)
    return directory / Path(str(max_epoch) + '_epochs.pth')


def get_arguments():
    parser = argparse.ArgumentParser(description="Process IF CSV files to identify each markers")
    parser.add_argument('--file', type = str, help = "Fichier contenant le model pré-entrainé")
    parser.add_argument('--ep', type = str, help = "Nombre d'époques")
    parser.add_argument('-n', action= 'store_true', help = "create a new model")

    args = parser.parse_args()

    return args 

def main(): 

    args = get_arguments()
        
    #number of markers 
    nb_in = 1
    nb_out = 1

    #Training parameters
    #batch_size = 16
    lr = 0.05 # Karpathy constant 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(nb_in,nb_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    loss_f = nn.MSELoss(reduction = "sum")

    path_trained_models = Path('./scripts/NN/nn_checkpoints/')

    # Bash arguments
    if args.n:
        epochs_done = 0
    else:
        if args.file:
            path_file = path_trained_models + args.file
            model, optimizer, epochs_done = load_nn(model,optimizer,path_file)
        else:
             path_file = most_trained_path(path_trained_models)
             model, optimizer, epochs_done = load_nn(model,optimizer,path_file)
    
    if args.ep:
        n_epochs = int(args.ep)
    else:
        n_epochs = 50

    
    checkpoint = train(X_test,Y_test,model,optimizer,loss_f,epochs_done,n_epochs)
    save_nn(checkpoint)


if (__name__== "__main__"):
    main()