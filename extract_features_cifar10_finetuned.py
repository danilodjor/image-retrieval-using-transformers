import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, vit_b_32, swin_b#, swin_v2_b

import pandas as pd
import numpy as np
import os
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

# Models & transforms (used for feature extraction with pre-trained models) setup

models = {"vit_b_16": vit_b_16,\
          "vit_b_32": vit_b_32}#,\
          #"swin_b": swin_b}
          #"swin_v2_b": swin_v2_b}

weights = {"vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,\
           "vit_b_32": torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1}#,\
            #"swin_b": torchvision.models.Swin_B_Weights.IMAGENET1K_V1}
            #"swin_v2_b": torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1}

img_transforms = {"vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms,\
              "vit_b_32": torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms}#,\
              #"swin_b": torchvision.models.Swin_B_Weights.IMAGENET1K_V1.transforms}
              #"swin_v2_b": torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms}

# Folder setup
train_features_file = "training_cifar10.pkl"
test_features_file = "test_cifar10.pkl"
features_folder = "./CIFAR-10/"
train_features_path = features_folder + train_features_file
test_features_path = features_folder + test_features_file

# Dataset setup
batch_size = 64 # was 16

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
        print(f'Reading existing {train_features_file}')
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

    # FEATURE EXTRACTION USING MODELS
    num_train_imgs = len(training_df)
    num_test_imgs = len(test_df)

    for model_name_base in models:
        model_name = model_name_base + '_finetuned'
        print('-----------------------------')
        print(f'Current model = {model_name}')

        if (model_name in training_df.columns) and (model_name in test_df.columns):
            continue

        model_weights = weights[model_name_base]
        model = models[model_name_base](weights = model_weights)
        model_save_path = f'./model_save/{model_name}.pth'
        model.load_state_dict(torch.load(model_save_path))
        model = model.to(device)
        model.eval()

        if model_name.startswith('swin'): # swin has different layer nomenclature
            num_features = model.head.in_features
            model.head = torch.nn.Identity()
        elif model_name.startswith('vit'): # all ViTs have the same layer nomenclature
            num_features = model.heads[0].in_features
            model.heads = torch.nn.Identity()

        model.eval()

        img_transform = img_transforms[model_name_base]()

        # Main feature extraction loop
        with torch.no_grad():
            # Feature extraction loop: Training set
            print(f'TRAINING SET EXTRACTION ({model_name}): ')
            model_features = np.zeros((num_train_imgs, num_features))
            for i in range(0, len(training_df), batch_size):
                images = torch.Tensor(np.stack(training_df.iloc[i:i+batch_size]['image']))
                images = images.to(device)
                images = img_transform(images)
                model_features[i:i+batch_size] = np.array(model(images).cpu())
                if i % 99 == 0:
                    print(f"{i+1}/{num_train_imgs}")

            training_df[model_name] = tuple(model_features)

            # Feature extraction loop: Test set
            print(f'TEST SET EXTRACTION ({model_name}): ')
            model_features = np.zeros((num_test_imgs, num_features))
            for i in range(0, len(test_df), batch_size):
                images = torch.Tensor(np.stack(test_df.iloc[i:i+batch_size]['image']))
                images = images.to(device)
                images = img_transform(images)
                model_features[i:i+batch_size] = np.array(model(images).cpu())
                if i % 99 == 0:
                    print(f"{i+1}/{num_test_imgs}")

            test_df[model_name] = tuple(model_features)

    # Saving the dataframes with extracted features
    training_df.to_pickle(train_features_path)
    test_df.to_pickle(test_features_path)

    print('#'*50 + '\n')
    print('Successfully saved the dataframes containing extracted features in pickle files.')

if __name__ == '__main__':
    main()
