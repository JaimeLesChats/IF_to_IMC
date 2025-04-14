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

import NN_model_trainer as mod

nb_in, nb_out = 1, 1
lr = 0.1

path_trained_models = Path('./scripts/NN/nn_checkpoints/')
path_file =  mod.most_trained_path(path_trained_models)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mod.SimpleNN(nb_in,nb_out).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)
model, _,_ = mod.load_nn(model,optimizer,path_file)


model.eval()
x_test = torch.tensor([[100.0], [-50.0], [1000.0]])
y_test_pred = model(x_test)
print(y_test_pred)