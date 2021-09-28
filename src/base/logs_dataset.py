from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import os
import torch
import numpy as np

def read_file(path):
  '''read hdfs files from disk 
  format is '1 2 3 4\t0'
  '''
  datas = []
  with open(path,'r') as f:
    for line in f:
      tmp = line.split("\t")
      datas.append((tmp[0],int(tmp[1])))
  return datas

def logkey_to_int(logkey):
    windows = 20
    tmp = logkey.split(" ")
    arr = [int(i) for i in tmp]
    if len(arr) > windows:
        arr = arr[0:windows]

    if len(arr) < windows :
        while len(arr)< windows:
            arr.append(-1)

    return arr


class LogDataset(Dataset):
    """
    LogDataset class for log dataset which preprocess by Drain3

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """


    def __init__(self, root: str, dataset_name: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        self.root = root
        self.dataset_name = dataset_name
        self.train = train  # training set or test set

        # load data
        train_file = read_file(root+'/'+dataset_name+'/train.txt')
        vaild_file = read_file(root+'/'+dataset_name+'/vaild.txt')
        test_file = read_file(root+'/'+dataset_name+'/test.txt')

        # 
        X_train = [ logkey_to_int(X) for X, y in train_file]
        X_test = [ logkey_to_int(X) for X, y in test_file]
        y_train = [ y for X, y in train_file]
        y_test = [y for X, y in test_file]


        if self.train:
            self.data = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)
