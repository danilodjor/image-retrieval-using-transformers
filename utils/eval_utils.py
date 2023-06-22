import pandas as pd
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
import cv2 as cv
import random
from matplotlib import pyplot as plt

from torchmetrics.functional import retrieval_average_precision, retrieval_recall, retrieval_precision
from torch.nn import CosineSimilarity

import json

def random_query(model_str, num_queries:int, topK:int, train_df:pd.DataFrame, test_df:pd.DataFrame, save_folder:str, device):
    num_images_test = len(test_df)

    sel_idx = random.sample(range(num_images_test), num_queries)

    big_grid = []

    database_matrix = torch.Tensor(train_df[model_str]).to(device)

    for i in sel_idx:
        # Extract a random test query image from the database
        query_img = test_df.iloc[i]['image']
        query_label = test_df.iloc[i]['label']
        query_ftrs = test_df.iloc[i][model_str]
        query_ftrs = torch.Tensor(query_ftrs).to(device) # transfer to CUDA for faster computations

        # Multiply the database matrix and the query feature vector (to be done for each query vector)
        # match_scores = database_matrix @ query_ftrs # inner product
        match_scores = CosineSimilarity()(database_matrix, query_ftrs) # cosine similarity
        sorted_scores, sorted_ind = torch.sort(match_scores, descending=True)

        # Take K best matches
        k_best_ind = sorted_ind[0:topK].cpu().numpy()

        # Grid for visualization
        grid_list = list(train_df.iloc[k_best_ind]['image'])

        query_img = cv.copyMakeBorder(query_img.transpose(1,2,0), 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=0)
        query_img = query_img.transpose(2,0,1)

        for j, img in enumerate(grid_list):
            img = img.transpose(1,2,0)
            
            color = (0,1,0) if query_label == train_df.iloc[k_best_ind[j]]['label'] else (1,0,0)
            img_with_border = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=color)

            grid_list[j] = torch.Tensor(img_with_border.transpose(2,0,1))

        grid_list = [torch.Tensor(query_img)] + grid_list # add query as first image
        grid = make_grid(grid_list, nrow=topK+1)

        big_grid.append(grid)

    plt.figure()
    big_grid = make_grid(big_grid, nrow=1, normalize=True)
    plt.imshow(big_grid.permute(1,2,0))
    plt.title(f'10 random queries and\n top 10 retrieved images [CIFAR-10, {model_str}]')
    plt.axis('off')
    plt.savefig(save_folder + f'CIFAR10_randQueries_{model_str}.png')


def get_metrics(model_str, train_df:pd.DataFrame, test_df:pd.DataFrame, K: list[int], save_folder:str, filename: str, device):
    sim_measure = CosineSimilarity()

    num_images_test = len(test_df)

    database_matrix = torch.Tensor(train_df[model_str]).to(device)
    database_labels = torch.Tensor(train_df['labels']).to(device)

    avg_prec = torch.zeros(num_images_test).to(device)
    precision = {k: torch.zeros(num_images_test).to(device) for k in K}
    recall = {k: torch.zeros(num_images_test).to(device) for k in K}

    for i in range(num_images_test):
        query_ftrs = test_df.iloc[i][model_str]
        query_ftrs = torch.Tensor(query_ftrs).to(device)

        query_label = test_df.iloc[i]['labels']
        
        targets = torch.where(database_labels == query_label, True, False).to(device)
        preds = sim_measure(database_matrix, query_ftrs) # cosine similarity

        avg_prec[i] = retrieval_average_precision(preds, targets)
        for k in K:
            precision[k][i] = retrieval_precision(preds, targets, k) # precision at k for image i
            recall[k][i] = retrieval_recall(preds, targets, k)
    
    avg_prec = avg_prec.cpu().numpy()
    for k in precision:
        precision[k] = precision[k].cpu().numpy()
    for k in recall:
        recall[k] = recall[k].cpu().numpy()
    
    # Plot the histogram of average precisions:
    plt.figure()
    plt.hist(avg_prec, bins=100, range=(0,1), edgecolor='k')
    plt.xlabel('Average Precision')
    plt.ylabel('Count')
    plt.title(f'Histogram of average precision values\n on the queries coming from the test set [CIFAR-10, {model_str}]')
    plt.axvline(avg_prec.mean(), color='r', linestyle='dashed', linewidth=1)

    min_ylim, max_ylim = plt.ylim()
    plt.text(avg_prec.mean()*0.65, max_ylim*0.9, 'Mean: {:.2f}'.format(avg_prec.mean()))
    plt.savefig(save_folder+f'CIFAR10_AP_hist_{model_str}.png', dpi=300)

    # Save the final evaluation result dictionary
    result_dict = {'model': model_str, 'AP': avg_prec, 'precision@k': precision, 'recall@k': recall}
    np.save(f'{filename}.npy', result_dict)