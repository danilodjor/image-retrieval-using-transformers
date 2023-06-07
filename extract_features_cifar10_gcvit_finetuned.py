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
train_features_file = "training_cifar10.pkl"
test_features_file = "test_cifar10.pkl"
features_folder = "./CIFAR-10/"
train_features_path = features_folder + train_features_file
test_features_path = features_folder + test_features_file
model_name = 'gc_vit_b_finetuned'
model_save_path = f'./model_save/{model_name}.pth'

# Dataset setup
batch_size = 1 # was 8

train_set = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_len = len(train_loader) # number of batches
test_len = len(test_loader)


# Main loop
def main():
    # Check that folders exist 
    if not os.path.exists(features_folder):
        raise Exception("Extracted features folder does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    # Models & transforms (used for feature extraction with pre-trained models) setup
    gc_model = timm.create_model('gcvit_base.in1k', checkpoint_path = model_save_path)
    gc_model.eval()

    gc_pool_layer = SelectAdaptivePool2d(pool_type='avg', flatten=Flatten(start_dim=1, end_dim=-1))
    gc_transform = timm.data.create_transform(
        **timm.data.resolve_data_config(gc_model.pretrained_cfg)
    )
    num_features = gc_model.head.fc.in_features

    models = ['gcvit_b_finetuned']
    gc_model = gc_model.to(device)

    # TRAINING: Initial dataframe
    if not os.path.exists(train_features_path):
        print('Initial train database save file in progress...')

        training_df = pd.DataFrame(columns=['image','label'])

        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = np.array(images)
            labels = np.array(labels)
            new_rows = pd.DataFrame({'image': tuple(images), 'label':  tuple(labels)})
            training_df = pd.concat([training_df, new_rows], ignore_index=True)
            if i%99 == 0:
                print(f"{i+1}/{len(train_loader)}")

        training_df.to_pickle(train_features_path)

        print('Initial train database save file done.')
    else:
        print(f'Reading existing initial training features file')
        training_df = pd.read_pickle(train_features_path)

    # TEST: Initial dataframe
    if not os.path.exists(test_features_path):
        print('Initial test database save file in progress...')

        test_df = pd.DataFrame(columns=['image','label'])
        for i, batch in enumerate(test_loader):
            images, labels = batch
            images = np.array(images)
            labels = np.array(labels)
            new_rows = pd.DataFrame({'image': tuple(images), 'label':  tuple(labels)})
            test_df = pd.concat([test_df, new_rows], ignore_index=True)
            if i%99 == 0:
                print(f"{i+1}/{len(test_loader)}")
        test_df.to_pickle(test_features_path)
        
        print('Initial test database save file done.')
    else:
        print(f'Reading existing {test_features_file}')
        test_df = pd.read_pickle(test_features_path)

    # FEATURE EXTRACTION USING GC-VIT
    num_train_imgs = len(training_df)
    num_test_imgs = len(test_df)

    for model_name in models:
        print('-----------------------------')
        print(f'Current model = {model_name}')

        if (model_name in training_df.columns) and (model_name in test_df.columns):
            continue

        # if model_name.startswith('swin'): # swin has different layer nomenclature
        #     num_features = model.head.in_features
        #     model.head = torch.nn.Identity()
        # elif model_name.startswith('vit'): # all ViTs have the same layer nomenclature
        #     num_features = model.heads[0].in_features
        #     model.heads = torch.nn.Identity()

        # Main feature extraction loop
        with torch.no_grad():
            print(f'Batch size = {batch_size}')
            # Feature extraction loop: Training set
            print(f'TRAINING SET EXTRACTION ({model_name}): ')
            model_features = np.zeros((num_train_imgs, num_features))
            for i in range(0, len(training_df)):
                images = torch.Tensor(training_df.iloc[i]['image'])

                images = transforms.ToPILImage()(images)
                images = gc_transform(images).unsqueeze(0)
                images = images.to(device)
                
                model_features[i:i+batch_size] = np.array(gc_pool_layer(gc_model.forward_features(images)).cpu())
                if i % 99 == 0:
                    print(f"{i+1}/{num_train_imgs}")

            training_df[model_name] = tuple(model_features)
            training_df.to_pickle(train_features_path)
            print(f'Saved {model_name} training features.')

            # Feature extraction loop: Test set
            print(f'TEST SET EXTRACTION ({model_name}): ')
            model_features = np.zeros((num_test_imgs, num_features))
            for i in range(0, len(test_df)):
                images = torch.Tensor(test_df.iloc[i]['image'])

                images = transforms.ToPILImage()(images)
                images = gc_transform(images).unsqueeze(0)
                images = images.to(device)
                
                model_features[i:i+batch_size] = np.array(gc_pool_layer(gc_model.forward_features(images)).cpu())
                if i % 99 == 0:
                    print(f"{i+1}/{num_test_imgs}")

            test_df[model_name] = tuple(model_features)
            test_df.to_pickle(test_features_path)
            print(f'Saved {model_name} test features.')

    print('#'*50 + '\n')
    print('Successfully saved the dataframes containing extracted features in pickle files.')

if __name__ == '__main__':
    main()