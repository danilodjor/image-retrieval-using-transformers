import pandas as pd
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
import cv2 as cv
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# Folder setup
train_features_file = "training_cifar10.pkl"
test_features_file = "test_cifar10.pkl"
features_folder = "./CIFAR-10/"
img_save_folder = "./CIFAR-10/"
train_features_path = features_folder + train_features_file
test_features_path = features_folder + test_features_file

# Models used to extract image features
models = ["vit_b_16", "vit_b_32", "swin_b"]

# Dataset-specific information
label_to_class = {0:'airplane',\
                  1:'automobile',\
                  2:'bird',\
                  3:'cat',\
                  4:'deer',\
                  5:'dog',\
                  6:'frog',\
                  7:'horse',\
                  8:'ship',\
                  9:'truck'}

# Load the datasets
database_train_df = pd.read_pickle(train_features_path)
database_test_df = pd.read_pickle(test_features_path)

num_images_train = len(database_train_df)
num_images_test = len(database_test_df)

# For each model
num_runs = 10
K = 10

sel_idx = random.sample(range(num_images_test), num_runs)
for model_str in models:
    big_grid = []

    # database_matrix = np.stack(database_train_df[model_str]) # or
    database_matrix = torch.Tensor(database_train_df[model_str]).to(device)

    for i in sel_idx:
        # Extract a random test query image from the database
        query_img = database_test_df.iloc[i]['image']
        query_label = database_test_df.iloc[i]['label']
        query_ftrs = database_test_df.iloc[i][model_str]
        query_ftrs = torch.Tensor(query_ftrs).to(device) # transfer to CUDA for faster computations

        # Multiply the database matrix and the query feature vector (to be done for each query vector)
        match_scores = database_matrix @ query_ftrs
        # sorted_scores, sorted_ind = torch.sort(torch.Tensor(match_scores), descending=True)
        sorted_scores, sorted_ind = torch.sort(match_scores, descending=True)

        # Take K best matches
        k_best_ind = sorted_ind[0:K].cpu().numpy()

        # Grid
        grid_list = list(database_train_df.iloc[k_best_ind]['image'])

        query_img = cv.copyMakeBorder(torch.tensor(query_img).permute(1,2,0).numpy(), 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=0)
        query_img = torch.tensor(query_img).permute(2,0,1)

        for j, img in enumerate(grid_list):
            img = torch.tensor(img).permute(1,2,0).numpy()
            
            color = (0,1,0) if query_label == database_train_df.iloc[k_best_ind[j]]['label'] else (1,0,0)
            img_with_border = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=color)

            grid_list[j] = torch.tensor(img_with_border).permute(2,0,1)

        grid_list = [query_img] + grid_list # add query as first image
        grid = make_grid(grid_list, nrow=K+1)

        big_grid.append(grid)

        # Save images to img_save_folder
        # img_path = img_save_folder + img_name + str(i) + '.png'
        # save_image(grid, img_path)

    big_grid = make_grid(big_grid, nrow=1, normalize=True)

    save_image(big_grid, img_save_folder + 'CIFAR10_randQueries_' + model_str + '.png')
    print('Successfully performed queries and saved images to ' + img_save_folder + '.')