import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

import pandas as pd
import matplotlib.pyplot as plt
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

# Parameters
run_on_server = True

models = {"vit_b_16": vit_b_16, "vit_b_32": vit_b_32}
weights = {"vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,\
           "vit_b_32": torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1}

train_features_file = "training_cifar10.pkl"
test_features_file = "test_cifar10.pkl"

# Don't change
if run_on_server:
    features_folder = "/usr/itetnas04/data-scratch-01/ddordevic/data/cluster_scripts/vit_copy/CIFAR-10/"
else:
    features_folder = "C:/Users/danil/Desktop/Master thesis/Code/msc-thesis/CIFAR-10/"

train_features_path = features_folder + train_features_file
test_features_path = features_folder + test_features_file

# Dataset preparation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 8

train_set = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

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

    # TRAINING
    if not os.path.exists(train_features_path):
        training_df = pd.DataFrame(columns=['image','label'])

        print('Initial train database save file in progress.')

        for i, batch in enumerate(train_loader):
            images, labels = batch
            new_rows = pd.DataFrame({'image': tuple(images), 'label':  tuple(labels)})
            training_df = pd.concat([training_df, new_rows], ignore_index=True)
            if i%99 == 0:
                print(f"{i}/{len(train_loader)}")

        training_df.to_pickle(train_features_path)

        print('Initial train database save file done.')
    else:
        print(f'Reading existing {train_features_file}')
        training_df = pd.read_pickle(train_features_path)

    # TEST
    if not os.path.exists(test_features_path):
        print('Initial test database save file in progress.')

        test_df = pd.DataFrame(columns=['image','label'])
        for i, batch in enumerate(test_loader):
            images, labels = batch
            new_rows = pd.DataFrame({'image': tuple(images), 'label':  tuple(labels)})
            test_df = pd.concat([test_df, new_rows], ignore_index=True)
            if i%99 == 0:
                print(f"{i}/{len(test_loader)}")
        test_df.to_pickle(test_features_path)
        
        print('Initial test database save file done.')
    else:
        print(f'Reading existing {test_features_file}')
        test_df = pd.read_pickle(test_features_path)

    # Iterate over the models to be used (key is model)
    num_train_imgs = len(training_df)
    num_test_imgs = len(test_df)

    for key in models.keys():
        print(f'Key = {key}')

        model_weights = weights[key]
        model_transform = model_weights.transforms()
        model = models[key](weights = model_weights)

        num_features = model.heads[0].in_features

        model.heads = torch.nn.Identity()
        model.eval()


        # Main feature extraction loop
        with torch.no_grad():
            # Feature extraction loop: Training set
            print(f'TRAINING ({key}): ')
            model_features = np.zeros((num_train_imgs, num_features))
            for i in range(0, len(training_df), batch_size):
                images = torch.stack(tuple(training_df.iloc[i:i+batch_size]['image']))
                images = images.to(device)
                model_features[i:i+batch_size] = model(model_transform(images))
                if i % 100 == 0:
                    print(f"{i+1}/{num_train_imgs}")

            training_df[key] = tuple(model_features)

            # Feature extraction loop: Test set
            print(f'TEST ({key}): ')
            model_features = np.zeros((num_test_imgs, num_features))
            for i in range(0, len(test_df), batch_size):
                images = torch.stack(tuple(test_df.iloc[i:i+batch_size]['image']))
                images = images.to(device)
                model_features[i:i+batch_size] = model(model_transform(images))
                if i % 100 == 0:
                    print(f"{i+1}/{num_test_imgs}")

            test_df[key] = tuple(model_features)

    # Saving the dataframes with extracted features
    training_df.to_pickle(train_features_path) #"/training_mnist.pkl")
    test_df.to_pickle(test_features_path) #"/test_mnist.pkl")

    print('Sucessfully saved the dataframes containing extracted features in pickle files.')

if __name__ == '__main__':
    main()
