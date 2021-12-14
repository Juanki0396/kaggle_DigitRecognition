
import os
import torch
import pandas as pd
import numpy as np
import tqdm
import argparse

from torch.utils.data import DataLoader
from src import dataset, ResNet

# Parsing arguments

my_parser = argparse.ArgumentParser(prog='inference',
                                    description='Make inference on the MNIST dataset examples.')

my_parser.add_argument('--csvPath',
                       help='Path from the test csv.',
                       default='data/test.csv')

my_parser.add_argument('--outputPath',
                       help='Directory to where the predictions will be stored.',
                       default='data/')

my_parser.add_argument('--modelPath',
                       help='Path from where the model parameters are loaded.',
                       default='models/Adam_ep_5_lr_0.001_batch_64/model')

args = my_parser.parse_args()

# Loading the model

modelPath = args.modelPath
model = ResNet.load_pretrained_model(ResNet.resnet30(), modelPath)
print(f'Model loaded succesfully.')

# Loading Test Dataset

datasetPath = args.csvPath
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

savePath = os.path.join(args.outputPath, 'predictions.csv')

df.to_csv(savePath, index=False)
