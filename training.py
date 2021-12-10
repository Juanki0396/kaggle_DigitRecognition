import os

import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader
from src import dataset, train, ResNet, data_preprocess


# Parsing arguments

my_parser = argparse.ArgumentParser(prog='training',
                                    description='Traininng a homemade ResNet model')

my_parser.add_argument('--path',
                       help='Path from the training csv.',
                       default='data/train.csv')

my_parser.add_argument('--outputPath',
                       help='Directory to save model and training plots',
                       default=None)

my_parser.add_argument('--learningRate',
                       help='Setting up learning rate',
                       type=float,
                       default=1e-3)

my_parser.add_argument('--epochs',
                       help='Setting up the number of training epochs',
                       type=int,
                       default=5)

my_parser.add_argument('--batchSize',
                       help='Setting up the batch size',
                       type=int,
                       default=64)

args = my_parser.parse_args()

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

epochs = args.epochs
lr = args.learningRate
batch_size = args.batchSize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossFunction = torch.nn.CrossEntropyLoss()

# Defining data paths, preprocessing it, creates dataset and creating data loaders

dataPath = args.path

print(f'Spliting the dataset from: {dataPath}')
training_df, test_df = data_preprocess.split_dataset(dataPath, train_ratio=0.9)

trainDataset = dataset.MNIST_TrainingDataset(dataframe=training_df)
testDatset = dataset.MNIST_TrainingDataset(dataframe=test_df)

trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
testDataLoader = DataLoader(testDatset, batch_size=1)

# Training step

losses = train.train_model(model, trainDataLoader, testDataLoader,
                           epochs, optimizer, lossFunction, lossFunction, device)

# Serializing the model

if args.outputPath is not None:
    saveDir = args.outputPath
else:
    saveDir = f'models/Adam_ep_{epochs}_lr_{lr}_batch_{batch_size}'

saveName = 'model'

if not os.path.exists(saveDir):
    os.mkdir(saveDir)

savePath = os.path.join(saveDir, saveName)

torch.save(model.state_dict(), savePath)

# Learning plots
# Training and testing epochs curve
steps = list(range(epochs))
average_train = losses['training_average']
average_test = losses['testing_average']

fig, axes = plt.subplots()
axes.plot(steps, average_train, label='Training')
axes.plot(steps, average_test, label='Testing')
axes.set_xlabel('Epochs')
axes.set_ylabel('Loss')
axes.set_title('Training_curve')
axes.legend()

figPath = os.path.join(saveDir, 'training_vs_testing.png')

fig.savefig(figPath)

# Learning Curve

train_loss = losses['training_batchs']
steps = list(range(len(train_loss)))

fig, axes = plt.subplots()
axes.plot(steps, train_loss, label='Training')
axes.set_xlabel('Steps')
axes.set_ylabel('Loss')
axes.set_title('Training_curve')

figPath = os.path.join(saveDir, 'learning_curve.png')

fig.savefig(figPath)

# Metric curve

steps = list(range(epochs))
train_loss = losses['metric_average']


fig, axes = plt.subplots()
axes.plot(steps, train_loss, label='Metric')
axes.set_xlabel('Epochs')
axes.set_ylabel('Metric')
axes.set_title('Metric_curve')

figPath = os.path.join(saveDir, 'metric_curve.png')

fig.savefig(figPath)
