
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from src import dataset, train, ResNet

# Creating the model

model = ResNet.resnet30()
print('ResNet Model Created')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Working device')

# Defining training hyperparameters

print('Setting up hyperparameters')

epochs = 100
lr = 1e-3
batch_size = 64
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossFunction = torch.nn.CrossEntropyLoss()


# Defining data paths and creating data loaders

trainPath = r'data/processed/train_dataset.csv'
testPath = r'data/processed/test_dataset.csv'

trainDataset = dataset.MNIST_TrainingDataset(trainPath)
testDatset = dataset.MNIST_TrainingDataset(testPath)

trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
testDataLoader = DataLoader(testDatset, batch_size=1)

# Training step

losses = train.train_model(model, trainDataLoader, testDataLoader,
                           epochs, optimizer, lossFunction, lossFunction, device)
