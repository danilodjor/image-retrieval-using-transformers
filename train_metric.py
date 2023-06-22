import timm
import torch
import argparse
import torch.optim as optim
from torchvision import datasets

from utils.metric_train_utils import *

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 0}")

# Main training loop:
models_to_train = ["vit_s16", "swin_s", "gcvit_s"] # list of models to be trained

models = {"vit_s16": 'vit_small_patch16_224.augreg_in1k',\
        "swin_s": 'swin_small_patch4_window7_224.ms_in1k',\
        "gcvit_s": 'gcvit_small.in1k'}

model_save_folder = './model_save/'

# Training hyperparameters
num_epochs = 10
batch_size = 32
lr = 3e-5
weight_decay = 5e-4

# Pytorch-metric-learning stuff
reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
reducer = MultipleReducers(reducer_dict)
distance = distances.CosineSimilarity()
loss_func = losses.ContrastiveLoss(pos_margin=1.0, neg_margin=0, distance=distance, reducer=reducer)
mining_func = miners.MultiSimilarityMiner(epsilon=0.1)
accuracy_calculator = AccuracyCalculator(include=("mean_average_precision",), k=128)

# Main
def main():
    for model_name in models_to_train:
        model_save_path = model_save_folder + f'{model_name}_ft_contr.pth'

        model = timm.create_model(models[model_name], pretrained=True)
        model = model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        
        # Dataset setup
        transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg, model=model))

        dataset1 = datasets.CIFAR10(root='../datasets/CIFAR-10', train=True, download=True, transform=transform)
        dataset2 = datasets.CIFAR10(root='../datasets/CIFAR-10', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Training loop
        print(f'Starting training of {model_name}:')
        for epoch in range(1, num_epochs + 1):
            train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        
        print(f'Training of {model_name} in {num_epochs} done. Saving...')
        torch.save(model.state_dict(), model_save_path)
        print(f'Saved {model_name}')

        print(f'Evaluating {model_name} on test set...')
        with torch.no_grad():
            test(dataset1, dataset2, model, accuracy_calculator)

        print(f'Finished evaluation of {model_name}.')

    print(f'All models trained, evaluated and saved!')

# Run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--models', nargs='+', required=False, help='Models to train: vit_s16, swin_s, gcvit_s')
    parser.add_argument('--lr', required=False, type=float, help='Learning rate')
    parser.add_argument('-b', '--batch_size', required=False, type=int, help='Batch size')
    parser.add_argument('-n', '--epochs', required=False, type=int, help='Number of epochs')

    args = parser.parse_args()

    if args.models:
        models_to_train = args.models
    if args.lr:
        lr = args.lr
    if args.batch_size:
        batch_size = args.batch_size
    if args.epochs:
        num_epochs = args.epochs

    main()