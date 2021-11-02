
from torch.utils.data import dataset
from torchvision import transforms
import torch
import numpy as np
import pandas as pd


class MNIST_TrainingDataset(dataset):
    """ 
    Dataset torch class that will contain data to train or test the CNN.
    """

    def __init__(self, csv_path, trans=None, isTest=False,  mean=None, std=None):
        """ 
        Create an instance of the dataset from a csv. If isTest is True, mean and std must be given in order to normalize the data.
        """

        reader = pd.read_csv(csv_path)
        X = reader[reader.columns[1:]].to_numpy(
            dtype='float').reshape((-1, 1, 28, 28))
        if isTest:
            self.mean = mean
            self.std = std
        else:
            self.mean = np.mean(X)
            self.std = np.std(X)
        self.X = torch.from_numpy(X)
        Y = reader['label'].to_numpy().reshape(-1, 1)
        self.Y = torch.from_numpy(Y)
        self.trans = trans

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        x = transforms.Normalize(self.mean, self.std)(x)
        if self.trans:
            x = self.trans(x)
        return x, y
