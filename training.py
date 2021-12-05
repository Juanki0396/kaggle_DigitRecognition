
import os
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

epochs = 5
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

# Serializing the model

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
