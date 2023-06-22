import pandas as pd
import numpy as np
import torch

def init_dataframe(dataloader: pd.DataFrame):
    """
    Takes a dataloader of images, extracts each image and label as numpy arrays,
    assigns them a unique id, and saves them inside a pandas DataFrame.
    """
    resulting_df = pd.DataFrame(columns=['image','label'])

    for i, batch in enumerate(dataloader):
        images, labels = batch
        images = np.array(images)
        labels = np.array(labels)
        new_rows = pd.DataFrame({'image': tuple(images), 'label':  tuple(labels)})
        resulting_df = pd.concat([resulting_df, new_rows], ignore_index=True)
        if i%99 == 0:
            print(f"{i+1}/{len(dataloader)}")
            
    resulting_df['img_id'] = np.arange(1, len(resulting_df)+1)

    return resulting_df

def extract_features(model, dataloader, device):
    features = torch.Tensor().to(device)
    labels_tensor = torch.Tensor().to(device)

    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        output = model(imgs)

        features = torch.concat([features, output])
        labels_tensor = torch.concat([labels_tensor, labels.to(device)])
        
        if i % 99 == 0:
            print(f"{i+1}/{len(dataloader)}")

    return features, labels_tensor 
