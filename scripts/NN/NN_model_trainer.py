import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import re
import csv
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm

import NN_dataset

# Get configuration file

def get_config():
    with open("./scripts/config.yaml",'r') as f:
        config = yaml.safe_load(f)
    return config


config = get_config()

# NEURAL NETWORK

class SimpleNN(nn.Module):
    def __init__(self, nb_in, nb_out, hidden_dim=1000):
        super().__init__()

        self.input_dim = nb_in
        self.output_dim = nb_out

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


# FUNCTIONS

def create_checkpoint(model, optimizer, epochs, validation_loss= None, training_loss = None):
    """
    Return a checkpoint in the training of the specified model

    - model : the neural network model
    - optimizer : the optimizer used
    - epochs : the total number of training epochs
    - validation_loss : current validation loss
    - training_loss : current training loss

    """
    checkpoint = {'epochs' : epochs,
                   'model_state_dict' : model.state_dict(),
                   'optimizer_state_dict' : optimizer.state_dict(),
                   'validation_loss' : validation_loss,
                   'training_loss' : training_loss,
                   'lr' : optimizer.param_groups[0]['lr'],
                   'input_dim' : model.input_dim,
                   'output_dim' : model.output_dim}
     
    return checkpoint


def save_nn(checkpoint):
    """
    Save in a .pth file the neural network model as a checkpoint
    
    - checkpoint : training checkpoint achieved by the model
    
    """
    directory = config['directory_trained_models']

    if not os.path.exists(directory):
            os.makedirs(directory)

    nn_basename = str(checkpoint['epochs']) + '_epochs.pth'
    nn_path = directory + nn_basename

    torch.save(checkpoint,nn_path)


def load_nn(nn_path, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Load the model, optimizer and the number of training epochs according to a saved model checkpoint
    
    - nn_path : path to the file of the saved checkpoint
    - device : device used to train the model
    
    """
    checkpoint = torch.load(nn_path)

    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']

    model = SimpleNN(input_dim,output_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr = checkpoint['lr']) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epochs']


def train(train_loader,val_loader,model,optimizer, epochs_done, epochs = 50, loss_f = nn.MSELoss(reduction = "sum")):
    """
    Trains a model according to some data
    
    - train_loader : training data, batch-sized
    - val_loader : validation data, batch-sized
    - model : neural network model
    - optimizer : optimizer
    - epochs_done : number of epochs already done before this training session
    - epochs : number of epochs of this training session
    - loss_f : loss function used for back-propagation

    Returns

    - checkpoint : checkpoint of the model after [epochs_done + epochs] epochs

    """

    loop = tqdm.tqdm(range(epochs_done+1,epochs_done+epochs+1),
                     ncols=100, colour='green', desc='Training from epoch {}'.format(epochs_done), )

    for epoch in loop:

        # Training
        model.train()
        for inputs, labels in train_loader:
            #Forward
            outputs = model(inputs)
            training_loss = loss_f(outputs,labels)

            #Backward
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
        
        #Validation
        model.eval()
        with torch.no_grad():
            for inputs,labels in val_loader:
                outputs = model(inputs)
                validation_loss = loss_f(outputs,labels)
                
        loop.set_postfix(training_loss=training_loss.item())
    
    checkpoint = create_checkpoint(model,optimizer,epochs_done+epochs,training_loss=training_loss, validation_loss = validation_loss)

    return checkpoint


def most_trained_path(directory):
    """
    Returns the most trained model of a directory
    
    - directory : directory where all trained models are
    
    """

    max_epoch = 0
    for filename in os.listdir(directory):
        epochs = int(re.split(r'_',filename)[0])
        max_epoch = max(max_epoch,epochs)
    return directory / Path(str(max_epoch) + '_epochs.pth')


def add_checkpoint_loss(epoch, training_loss, validation_loss):
    """
    Writes in a csv files the losses of a model corresponding to the epoch of training
    
    - epoch : epochs done so far
    - training_loss : training loss
    - validation_loss : validation loss
    
    """

    with open(config['path_training_logs'], "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch,float(training_loss), float(validation_loss)])


def get_arguments():
    """
    Get arguments when launching the python script
    
    """
    
    parser = argparse.ArgumentParser(description="Process IF CSV files to identify each markers")
    parser.add_argument('--file', type = str, help = "Fichier contenant le model pré-entrainé")
    parser.add_argument('--ep', type = str, help = "Nombre d'époques")
    parser.add_argument('-n', action= 'store_true', help = "create a new model")

    args = parser.parse_args()

    return args 


def main(): 

    args = get_arguments()

    loss_f = loss_f = nn.MSELoss(reduction = "sum")

    train_loader, val_loader = NN_dataset.get_dataset()

    # Bash arguments
    if args.n:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = SimpleNN(config['input_dim'],config['output_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'])
        epochs_done = 0

    else:
        path_trained_models = Path(config['directory_trained_models'])

        if args.file:
            path_file = path_trained_models + args.file
            model, optimizer, epochs_done = load_nn(path_file)
        else:
             path_file = most_trained_path(path_trained_models)
             model, optimizer, epochs_done = load_nn(path_file)
    
    if args.ep:
        n_epochs = int(args.ep)
    else:
        n_epochs = 50 # Default number of epochs

    
    checkpoint = train(train_loader,val_loader,model,optimizer,epochs_done,n_epochs,loss_f=loss_f)
    save_nn(checkpoint)
    add_checkpoint_loss(checkpoint['epochs'],checkpoint['training_loss'],checkpoint['validation_loss'])

if (__name__== "__main__"):
    main()