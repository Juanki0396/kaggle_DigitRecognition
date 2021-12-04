
from torch.utils.data import dataset
from torchvision import transforms
import torch
import numpy as np
import pandas as pd


class MNIST_TrainingDataset(dataset):
    """ 
    Dataset torch class that will contain data to train or test the CNN.
    """

    def __init__(self, csv_path):
        """ 
        Create an instance of the dataset from a csv.
        """

        reader = pd.read_csv(csv_path)
        X = reader[reader.columns[1:]].to_numpy(
            dtype='float').reshape((-1, 1, 28, 28))
        self.X = torch.from_numpy(X)
        Y = reader['label'].to_numpy().reshape(-1, 1)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        x = transforms.Normalize(0, 1)(x)
        return x, y
