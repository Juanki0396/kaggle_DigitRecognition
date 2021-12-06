
# TODO Import pretrained model, get predictions over the test.csv, save results in correct format

import os
import torch
import pandas as pd
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from src import dataset, ResNet

# Loading the model

modelname = 'Adam_ep_5_lr_0.001_batch_64'
savePath = os.path.join('models', modelname, 'model')
model = ResNet.load_pretrained_model(ResNet.resnet30(), savePath)
print(f'Model loaded: {modelname} ')

# Loading Test Dataset

datasetPath = r'data/test.csv'
dataset = dataset.MNIST_TrainingDataset(datasetPath, isKaggleTest=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Making inference

results = []
id = 1
print('Starting inference')

with torch.no_grad():
    for X, _ in tqdm.tqdm(dataloader, desc='Inference'):
        pred = model(X)
        pred = np.argmax(pred.numpy().reshape(-1))
        results.append([id, pred])
        id += 1

# Saving results in a csv

df = pd.DataFrame(results, columns=['ImageId', 'Label'])
savePath = os.path.join('models', modelname, 'kaggle_submission.csv')
df.to_csv(savePath, index=False)
