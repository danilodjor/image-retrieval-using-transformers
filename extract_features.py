import pandas as pd
import numpy as np
from vit import *

""" 
Extracts the features of the images in the MNIST dataset using the trained vision transformer.
This way the database of MNIST images becomes a database of MNIST trasnformer feature representations.

Steps:
    1. Load the trained transformer model
    2. Strip off the final classification layer
    3. Go through images and forward propagate them
    4. For each image save the final transformer layer representation of that image.
"""

model_file = "vit_model_0404.pt"
model_folder = "/usr/itetnas04/data-scratch-01/ddordevic/data/cluster_scripts/vit_copy/model_save/"
model_path = model_folder + model_file

  
def main():
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
        for batch in tqdm(train_loader, leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x_features = model(x)
            new_row = {'image': x.flatten(), 'feature': x_features, 'label': y}
            training_df.append(new_row, ignore_index=True)

        # Feature extraction loop: Test set
        for batch in tqdm(test_loader, leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x_features = model(x)
            new_row = {'image': x.flatten(), 'feature': x_features, 'label': y}
            test_df.append(new_row, ignore_index=True)

    # Saving the dataframes with extracted features
    save_folder = "/usr/itetnas04/data-scratch-01/ddordevic/data/cluster_scripts/vit_copy/extracted_features"
    training_df.to_pickle(save_folder + "/training_mnist.pkl")
    test_df.to_pickle(save_folder + "/test_mnist.pkl")

    print('Sucessfully saved the dataframes containing extracted features in pickle files.')

if __name__ == '__main__':
    main()
