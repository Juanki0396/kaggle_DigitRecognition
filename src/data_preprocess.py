
from os import path
import pandas as pd
import random as rand
from torch.utils import data

from torch.utils.data import dataset


def split_dataset(dataset_path, train_ratio=0.95, seed=40):
    """ 
    Split the training set given by kaggle into a train and test set. 
    """

    # Fix the random seed
    rand.seed(seed)

    # Reading original file
    print('Reading CSV file.')
    dataset = pd.read_csv(dataset_path)

    # Obtaining the number of training examples
    n_train = int(len(dataset) * train_ratio)
    print(f'Number of training examples: {n_train}')
    print(f'Number of testing examples: {len(dataset) - n_train}')

    # Shuffling index list and split the list into training index and testing index
    idx = list(range(len(dataset)))
    rand.shuffle(idx)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    print('Dataset splitted randomly.')

    # returning splitted datasets
    return dataset.iloc[train_idx], dataset.iloc[test_idx]
