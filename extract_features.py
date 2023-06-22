import timm
from pprint import pprint

from timm.layers import SelectAdaptivePool2d
from torch.nn import Flatten

import torch
import torchvision
from torchvision import transforms

import os
import numpy as np
import pandas as pd

import argparse
from utils.extract_utils import *

""" 
Extracts the features of the images in the CIFAR-10 dataset using the pre-trained vision transformers.
This way the database of CIFAR-10 images becomes a database of CIFAR-10 transformer feature representations.
Python notebook is used to test the code before running it fully as a pure Python script. 

Steps:
    1. Load the trained transformer model
    2. Strip off the final classification layer
    3. Go through images and forward propagate them
    4. For each image save the final transformer layer representation of that image.
"""

# Folder setup
models = {"vit_s16": 'vit_small_patch16_224.augreg_in1k',\
          "swin_s": 'swin_small_patch4_window7_224.ms_in1k',\
          "gcvit_s": 'gcvit_small.in1k'}

dataset = 'cifar10'
method = 'contr'

# Dataset setup
batch_size = 32 # was 8

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Main loop
def main(model_name, weights):
    # Check that folders exist 
    if not os.path.exists(features_folder):
        raise Exception("Extracted features folder does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    # Models & transforms (used for feature extraction with pre-trained models) setup
    if weights is not None:
        model = timm.create_model(models[model_name], checkpoint_path = weights)
    else:
        model = timm.create_model(models[model_name])
    model.reset_classifier(0)
    model.eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    model = model.to(device)

    train_set = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print('-----------------------------')
    print(f'Current model = {model_name}')

    # Main feature extraction loop
    with torch.no_grad():
        print(f'TRAINING SET EXTRACTION ({model_name}): ')
        train_features, train_labels = extract_features(model, train_loader, device)
        train_df = pd.DataFrame({model_name: train_features.detach().cpu().numpy().tolist(), 'labels': train_labels.detach().cpu().numpy()})

        print(f'TEST SET EXTRACTION ({model_name}): ')
        test_features, test_labels = extract_features(model, test_loader, device)
        test_df = pd.DataFrame({model_name: test_features.detach().cpu().numpy().tolist(), 'labels': test_labels.detach().cpu().numpy()})

        train_df.to_pickle(train_features_path)
        test_df.to_pickle(test_features_path)
        print(f'Saved {model_name} test features.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', required=True, help='Pretrained model name (vit_s16, swin_s, gcvit_s)')
    parser.add_argument('-w', '--weights', required=False, help='Path to the finetuned weights in the same order as the models.')
    args = parser.parse_args()

    model = args.model
    weights = args.weights

    train_features_file = '_'.join([model, dataset, 'train', method])
    test_features_file = '_'.join([model, dataset, 'test', method])
    features_folder = "./CIFAR-10/"
    train_features_path = features_folder + train_features_file
    test_features_path = features_folder + test_features_file

    main(model, weights)