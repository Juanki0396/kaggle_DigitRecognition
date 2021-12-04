
import os
import pandas as pd

from src.data_preprocess import split_dataset

# Defining data directory and dataset path
data_dir = 'data/'

data_file = 'train.csv'
data_path = os.path.join(data_dir, data_file)

# Spliting the dataset randomly
print(f'Spliting the dataset from: {data_path}')
training_df, test_df = split_dataset(data_path, train_ratio=0.9)

# Defining and creating save directory
save_dir = os.path.join(data_dir, 'processed')

if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

# Saving files
train_dataset_path = os.path.join(save_dir, 'train_dataset.csv')
training_df.to_csv(train_dataset_path, index=False)

test_dataset_path = os.path.join(save_dir, 'test_dataset.csv')
test_df.to_csv(test_dataset_path, index=False)
