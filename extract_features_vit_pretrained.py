import pandas as pd
import numpy as np
import os
from torchvision.models import vit_b_1, swin_t, swin_v2_t
from vit import *

""" 
NOTE: Currently same as "extract_features.py" Need to experiment with extract_features_prelim.ipynb before doing this
Extracts the features of the images in the MNIST dataset using the trained vision transformer.
This way the database of MNIST images becomes a database of MNIST trasnformer feature representations.

Steps:
    1. Load the trained transformer model
    2. Strip off the final classification layer
    3. Go through images and forward propagate them
    4. For each image save the final transformer layer representation of that image.
"""

# Parameters:
run_on_server = True 

model_file = "vit_model_0404.pt"
train_features_file = "training_mnist.pkl"
test_features_file = "test_mnist.pkl"

if run_on_server:
    model_folder =  "/usr/itetnas04/data-scratch-01/ddordevic/data/cluster_scripts/vit_copy/model_save/"
    features_folder = "/usr/itetnas04/data-scratch-01/ddordevic/data/cluster_scripts/vit_copy/extracted_features/"
else:
    model_folder =  "C:/Users/danil/Desktop/Master thesis/Code/msc-thesis/model_save/"
    features_folder = "C:/Users/danil/Desktop/Master thesis/Code/msc-thesis/extracted_features/"


# Code (not supposed to be modified):
model_path = model_folder + model_file
train_features_path = features_folder + train_features_file
test_features_path = features_folder + test_features_file

  
def main():
    # Check that folders exist
    if not os.path.exists(model_folder):
        raise Exception("Model folder does not exist.")
    if not os.path.exists(features_folder):
        raise Exception("Extracted features folder does not exist.")

    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    # Loading the model: Replace the final layer with identity -> Keeps the features
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.mlp = nn.Identity()
    model.eval()

    # Pandas dataframe containing flattened images, their corresponding features and labels
    training_df = pd.DataFrame(columns=['image','feature','label'])
    test_df = pd.DataFrame(columns=['image','feature','label'])

    with torch.no_grad():
        # Feature extraction loop: Training set
        # i = 0
        for batch in tqdm(train_loader, leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x_features = model(x)
            new_row = pd.DataFrame([{'image': x[0].cpu(), 'feature': x_features[0].cpu(), 'label': y[0].cpu()}]) # saves an image CxHxW, and features
            training_df = pd.concat([training_df, new_row], ignore_index=True)
            # i+=1
            # if i == 5:
            #     break

        # Feature extraction loop: Test set
        # i = 0
        for batch in tqdm(test_loader, leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x_features = model(x)
            new_row = pd.DataFrame([{'image': x[0].cpu(), 'feature': x_features[0].cpu(), 'label': y[0].cpu()}])
            test_df = pd.concat([test_df, new_row], ignore_index=True)
            # i += 1
            # if i == 5:
            #     break

    # Saving the dataframes with extracted features
    training_df.to_pickle(train_features_path) #"/training_mnist.pkl")
    test_df.to_pickle(test_features_path) #"/test_mnist.pkl")

    print('Sucessfully saved the dataframes containing extracted features in pickle files.')

if __name__ == '__main__':
    main()
