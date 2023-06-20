import pandas as pd
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
import cv2 as cv
import random
from matplotlib import pyplot as plt

from torchmetrics import RetrievalMAP
from torchmetrics.functional import retrieval_average_precision, retrieval_recall, retrieval_precision
from torch.nn import CosineSimilarity

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
# models = ["vit_b_16", "vit_b_32", "swin_b", "vit_b_16_finetuned", "vit_b_32_finetuned", "swin_b_finetuned"]
models = ["gc_vit_b"]

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

class_to_label = {'airplane':0,\
                  'automobile':1,\
                  'bird':2,\
                  'cat':3,\
                  'deer':4,\
                  'dog':5,\
                  'frog':6,\
                  'horse':7,\
                  'ship':8,\
                  'truck':9}

classes = {'airplane',\
                  'automobile',\
                  'bird',\
                  'cat',\
                  'deer',\
                  'dog',\
                  'frog',\
                  'horse'\
                  'ship'\
                  'truck'}

# For each model
def random_query(models, num_queries:int, topK:int, train_df:pd.DataFrame, test_df:pd.DataFrame, save_folder:str, save_imgs:bool=True):
    num_images_test = len(test_df)

    sel_idx = random.sample(range(num_images_test), num_queries)

    for model_str in models:
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

        if save_imgs:
            plt.figure()
            big_grid = make_grid(big_grid, nrow=1, normalize=True)
            plt.imshow(big_grid.permute(1,2,0))
            plt.title(f'10 random queries and\n top 10 retrieved images [CIFAR-10, {model_str}]')
            plt.axis('off')
            plt.savefig(img_save_folder + f'CIFAR10_randQueries_{model_str}.png')
            # save_image(big_grid, save_folder + 'CIFAR10_randQueries_' + model_str + '.png')

# def cosine_similarity(A,B=None):
#     """
#     Calculates the cosine similiarites between rows of matrix A and matrix B.
#     Equivalent to sklearn.metrics.pairwise.cosine_similarity, but faster.
#     If B is empty, then the pairwise cosine similarities between rows of A will be used.
#     """
#     Anorm = A / np.linalg.norm(A, axis=-1)[:, np.newaxis]
#     if B is None:
#         Bnorm = Anorm
#     else:
#         Bnorm = B / np.linalg.norm(B, axis=-1)[:, np.newaxis]
#     return np.dot(Anorm, Bnorm.T)

def get_ap(models, train_df:pd.DataFrame, test_df:pd.DataFrame, save_folder:str, save_results=True):

    num_images_test = len(test_df)
    results_ap = {model_str: False for model_str in models}
    results_p = {model_str: False for model_str in models}
    results_r = {model_str: False for model_str in models}

    ap_classwise = {model_str: {label: [] for label in label_to_class.keys()} for model_str in models}

    for model_str in models:
        database_matrix = torch.Tensor(train_df[model_str]).to(device)
        ap_test = torch.zeros(num_images_test).to(device)
        p_test = torch.zeros(num_images_test).to(device)
        r_test = torch.zeros(num_images_test).to(device)

        database_labels = torch.Tensor(train_df['label']).to(device)

        for i in range(num_images_test):
            query_ftrs = test_df.iloc[i][model_str]
            query_ftrs = torch.Tensor(query_ftrs).to(device)

            # query_label = torch.Tensor(test_df.iloc[i]['label']).to(device)
            query_label = test_df.iloc[i]['label']
            
            target = torch.where(database_labels == query_label, True, False)

            # Multiply the database matrix and the query feature vector
            # preds = database_matrix @ query_ftrs # preds = similarity scores # inner product
            preds = CosineSimilarity()(database_matrix, query_ftrs) # cosine similarity

            ap_test[i] = retrieval_average_precision(preds, target)
            p_test[i] = retrieval_precision(preds, target, k=10)
            r_test[i] = retrieval_recall(preds, target, k=10)

            ap_classwise[model_str][query_label].append(ap_test[i].cpu())
        
        results_ap[model_str] = ap_test.cpu().numpy()
        results_p[model_str] = p_test.cpu().numpy()
        results_r[model_str] = r_test.cpu().numpy()

        ap_test = ap_test.cpu()
        
        # Plot the histogram of average precisions:
        if save_results:
            plt.figure()
            plt.hist(ap_test.numpy(), bins=100, range=(0,1), edgecolor='k')
            plt.xlabel('Average Precision')
            plt.ylabel('Count')
            plt.title(f'Histogram of average precision values\n on the queries coming from the test set [CIFAR-10, {model_str}]')
            plt.axvline(ap_test.mean(), color='r', linestyle='dashed', linewidth=1)

            min_ylim, max_ylim = plt.ylim()
            plt.text(ap_test.mean()*0.65, max_ylim*0.9, 'Mean: {:.2f}'.format(ap_test.mean()))
            plt.savefig(img_save_folder+f'CIFAR10_AP_hist_{model_str}.png', dpi=300)

        # Plot the AP histograms for each class separately
        fig, axs = plt.subplots(2, 5)
        fig.suptitle(f'APs of CIFAR10 classes, {model_str}')
        row=-1
        col=0
        for i, class_label in enumerate(label_to_class.keys()):
            if i%5==0:
                row+=1
                col=0

            axs[row, col].hist(np.array(ap_classwise[model_str][class_label]), bins=20, range=(0,1))
            axs[row, col].set_title(f'{label_to_class[class_label]}')

            col+=1
        fig.tight_layout(pad=1.0)
        plt.savefig(img_save_folder + f"CIFAR10_AP_hist_classwise_{model_str}", dpi=300)
 
    if save_results:
        torch.save(results_ap, save_folder + 'CIFAR10_models_APs.pkl')

    
def main():
    # Load the datasets
    database_train_df = pd.read_pickle(train_features_path)
    database_test_df = pd.read_pickle(test_features_path)
    
    # Perform random querying
    print("Performing random querying...")

    random_query(models=models,\
                 num_queries=10, topK=10,\
                 train_df=database_train_df,\
                 test_df=database_test_df,\
                 save_folder=img_save_folder,\
                 save_imgs=True)

    print(f'Successfully performed queries and saved images to {img_save_folder}.')

    # Draw histograms of results
    print('Getting average precision statistics...')

    get_ap(models=models,\
           train_df=database_train_df,\
           test_df=database_test_df,\
           save_folder=img_save_folder,\
           save_results=True)
    
    print(f'Successfully calculated average precision statistics and saved them to {img_save_folder}')

if __name__ == '__main__':
    main()