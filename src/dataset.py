
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import pandas as pd


class MNIST_TrainingDataset(Dataset):
    """ 
    Dataset torch class that will contain data to train or test the CNN.
    """

    def __init__(self, csv_path=None, dataframe=None, isKaggleTest=False):
        """ 
        Create an instance of the dataset from a csv.
        """
        if csv_path is not None:
            reader = pd.read_csv(csv_path)
        elif dataframe is not None:
            reader = dataframe
        if isKaggleTest:
            X = reader[reader.columns[:]].to_numpy(
                dtype='float').reshape((-1, 1, 28, 28))
            self.X = torch.from_numpy(X).to(dtype=torch.float32)
            Y = np.zeros((X.shape[0]))
            self.Y = torch.from_numpy(Y).to(dtype=torch.long)
        else:
            X = reader[reader.columns[1:]].to_numpy(
                dtype='float').reshape((-1, 1, 28, 28))
            self.X = torch.from_numpy(X).to(dtype=torch.float32)
            Y = reader['label'].to_numpy().reshape(-1)
            self.Y = torch.from_numpy(Y).to(dtype=torch.long)
        print(f'Created dataset with {len(self)} examples.')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        x = transforms.Normalize(0, 1)(x)
        return x, y
