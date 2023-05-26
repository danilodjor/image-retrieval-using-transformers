import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
import torchvision
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, vit_b_32, swin_b

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# Functions used in training
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
### pytorch-metric-learning stuff ###

# Model setup
models = {"vit_b_16": vit_b_16,\
        "vit_b_32": vit_b_32,\
        "swin_b": swin_b}
        #"swin_v2_b": swin_v2_b}

weights = {"vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,\
            "vit_b_32": torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1,\
            "swin_b": torchvision.models.Swin_B_Weights.IMAGENET1K_V1}
            #"swin_v2_b": torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1}

img_transforms = {"vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms,\
            "vit_b_32": torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms,\
            "swin_b": torchvision.models.Swin_B_Weights.IMAGENET1K_V1.transforms}
            #"swin_v2_b": torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms}

# Main training loop:
model_name_list = ["swin_b"]
for model_name in model_name_list:
    model_save_path = f'./model_save/{model_name}_finetuned.pth'

    model_weights = weights[model_name]
    model = models[model_name](weights = model_weights).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    # Dataset setup
    transform = img_transforms[model_name]()
    batch_size = 64

    dataset1 = datasets.CIFAR10(root='../datasets/CIFAR-10', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset2 = datasets.CIFAR10(root='../datasets/CIFAR-10', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Training loop
    print(f'Starting training of {model_name}:')
    for epoch in range(1, num_epochs + 1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    
    print(f'Training of {model_name} in {num_epochs} done.')
    print(f'Saving model {model_name}...')
    torch.save(model.state_dict(), model_save_path)
    print(f'Saved {model_name}')

    print(f'Evaluating {model_name} on test set...')
    with torch.no_grad():
        test(dataset1, dataset2, model, accuracy_calculator)

    print(f'Finished evaluation of {model_name}.')

print(f'All models trained, evaluated and saved is done!')