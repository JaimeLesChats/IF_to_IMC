import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader, random_split


#Dataset
class Dataset(Dataset):
    def __init__(self):
        X_dataset= torch.linspace(-10, 10, 100).view(-1, 1)
        Y_dataset= X_dataset*2
        self.X = torch.tensor(X_dataset)
        self.Y = torch.tensor(Y_dataset)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataset():

    dataset = Dataset()
  
    train_ratio = 0.8
    train_size = int(train_ratio*len(dataset))
    val_size = len(dataset) - train_size
    batch_size = 32

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)
    
    return train_loader, val_loader