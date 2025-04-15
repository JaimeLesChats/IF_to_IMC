import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import re
import matplotlib.pyplot as plt
import yaml

import torch

import NN_model_trainer as mod

#Config file

def get_config():
    with open("./scripts/config.yaml",'r') as f:
        config = yaml.safe_load(f)
    return config


config = get_config()

## 

def evaluate(trained_model_path):
    
    model, _,_ = mod.load_nn(trained_model_path)

    model.eval()
    x_test = torch.tensor([[100.0], [-50.0], [1000.0]])
    y_test_pred = model(x_test)
    print(y_test_pred)

def graph():
    df = pd.read_csv(config['path_training_logs'])
    epochs = df.iloc[:,0]
    train_loss = df.iloc[:,1]
    val_loss = df.iloc[:,2]

    plt.plot(epochs, train_loss, label = 'train_loss', color = 'blue')
    plt.plot(epochs, val_loss, label = 'val_loss', color = 'red')

    plt.show()


def get_best_model_path():

    df = pd.read_csv(config['path_training_logs'])

    min_row = df.iloc[df.iloc[:,2].idxmin()]
    best_epoch = min_row[0]

    return Path(config['directory_trained_models'] + str(int(best_epoch)) + '_epochs.pth')

evaluate(get_best_model_path())

