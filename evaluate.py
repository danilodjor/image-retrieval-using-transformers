import pandas as pd
import torch
import argparse

from utils.eval_utils import *

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# Folder setup
dataset = 'cifar10'
method = 'contr'

def main(model, train_path, test_path, K, savedir):
    # Load the datasets
    database_train_df = pd.read_pickle(train_path)
    database_test_df = pd.read_pickle(test_path)
    
    # Perform random querying
    # print("Performing random querying...")

    # random_query(model_str=model,\
    #              num_queries=10, topK=10,\
    #              train_df=database_train_df,\
    #              test_df=database_test_df,\
    #              save_folder=savedir,\
    #              device=device)

    print(f'Successfully performed queries and saved images to {savedir}.')

    # Draw histograms of results
    print('Getting average precision statistics...')

    get_metrics(model_str=model,\
           train_df=database_train_df,\
           test_df=database_test_df,\
           K=K,\
           filename=filename,\
           save_folder=savedir,\
           device=device)
    
    print(f'Successfully calculated average precision statistics and saved them to {savedir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Name of the model used for feature extraction (vit_s16, swin_s, gcvit_s).')
    parser.add_argument('--train', required=True, help='Path to the file with database (training) set features extracted.')
    parser.add_argument('--test', required=True, help='Path to the file with query (test) set features extracted.')
    parser.add_argument('--K', required=True, nargs='+', type=int, help='Path to the file with query (test) set features extracted.')
    parser.add_argument('--savedir', required=True, help='Directory in which the results will be saved.')
    
    args = parser.parse_args()

    model = args.model
    train_path = args.train
    test_path = args.test
    savedir = args.savedir
    K = args.K

    filename = f"{model}_{dataset}_{method}_metrics.json"

    main(model, train_path, test_path, K, savedir)