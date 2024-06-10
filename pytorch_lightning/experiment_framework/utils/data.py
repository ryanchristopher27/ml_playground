"""
Purpose: Data Tools
"""

# Imports
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import scale, MinMaxScaler
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import math
import lightning as L

class Dataset:
    """
    Purpose: Machine learning dataset
    """
    
    def __init__(self, samples, labels, shuffle=False):
        """
        Purpose: 
        - Define instance parameters, optionally shuffle

        Arguments:
        - samples (list): sequence of dataset observations
        - labels (list): dataset labels formatted as scalar or sequence
        """
        
        self.samples = samples
        self.labels = labels

        if shuffle:
            
            indices = np.arange(self.samples.shape[0])
            np.random.shuffle(indices)
            
            self.samples = self.samples[indices]
            self.labels = self.labels[indices]

    def __getitem__(self, index):
        """
        Purpose: 
        - Get dataset information for given index iteration

        Arguments:
        - index (int): current interation counter

        Returns:
        - (tuple[any]): sample and label that corresponds to iteration index
        """
        
        sample, label = self.samples[index], self.labels[index]

        return (sample.astype(np.float32), label.astype(np.float32))

    def __len__(self):
        """
        Purpose: 
        - Get number of dataset observations

        Returns:
        - (int): number of dataset observations
        """

        return self.samples.shape[0]



class StockDataModule(L.LightningModule):
    def __init__(self, batch_size, train_dataset, val_dataset, test_dataset):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def setup(self, stage=None):
        None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def create_stock_dataset(num_features = 1, seq_len = 50, train_scaler = MinMaxScaler(feature_range=(0, 1)), test_scaler = MinMaxScaler(feature_range=(0, 1))):
    print("Data Preprocessing")
    # print(f"Num Features: {num_features}")
    # print(f"Sequence Length: {seq_len}")
    
    start_date = dt.datetime(2020,4,1)
    end_date = dt.datetime(2024,4,1)
 
    #loading from yahoo finance
    data = yf.download("AAPL",start_date, end_date)

    training_data_len = math.ceil(len(data) * .8) 

    if num_features == 1:
        # Grab only the Open Price
        train_data = data[:training_data_len].iloc[:,:1] # shape = (605, 1)
        test_data = data[training_data_len:].iloc[:,:1] # shape = (151, 1)
        dataset_train = train_data.Open.values 
        # Reshaping 1D to 2D array
        dataset_train = np.reshape(dataset_train, (-1,1)) # shape = (605, 1)
        # scaler = MinMaxScaler(feature_range=(0,1))
        scaled_train = train_scaler.fit_transform(dataset_train)
        # Selecting Open Price values
        dataset_test = test_data.Open.values 
        # Reshaping 1D to 2D array
        dataset_test = np.reshape(dataset_test, (-1,1)) 
        # Normalizing values between 0 and 1
        scaled_test = test_scaler.fit_transform(dataset_test) 
        # Create sequence of length [seq_len]
        # seq_len = 50
        X_train = []
        y_train = []
        for i in range(seq_len, len(scaled_train)):
            X_train.append(scaled_train[i-seq_len:i, 0])
            y_train.append(scaled_train[i, 0])
        X_test = []
        y_test = []
        for i in range(seq_len, len(scaled_test)):
            X_test.append(scaled_test[i-seq_len:i, 0])
            y_test.append(scaled_test[i, 0])
        # The data is converted to Numpy array
        X_train, y_train = np.array(X_train), np.array(y_train)
        #Reshaping
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
        y_train = np.reshape(y_train, (y_train.shape[0],1))
        # print("X_train :",X_train.shape,"y_train :",y_train.shape)
        # The data is converted to numpy array
        X_test, y_test = np.array(X_test), np.array(y_test)
        #Reshaping
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
        y_test = np.reshape(y_test, (y_test.shape[0],1))
    else:
        train_data = data[:training_data_len].iloc[:,:] 
        test_data = data[training_data_len:].iloc[:,:]

        dataset_train = train_data.values
        # scaler = MinMaxScaler(feature_range=(0,1))
        dataset_train = train_scaler.fit_transform(dataset_train)

        dataset_train = np.reshape(dataset_train, (-1,6)) 
        scaled_train = dataset_train.reshape((dataset_train.shape[0], 6, 1))

        dataset_test = test_data.values  
        scaled_test = test_scaler.fit_transform(dataset_test) 
        scaled_test = np.reshape(scaled_test, (-1,6)) 
        scaled_test = scaled_test.reshape((scaled_test.shape[0], 6, 1))

        X_train = []
        y_train = []
        for i in range(seq_len, len(scaled_train)):
            X_train.append(scaled_train[i-seq_len:i, :, 0])
            y_train.append(scaled_train[i, 0])

        X_test = []
        y_test = []
        for i in range(seq_len, len(scaled_test)):
            X_test.append(scaled_test[i-seq_len:i, :, 0])
            y_test.append(scaled_test[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        y_train = np.reshape(y_train, (y_train.shape[0],1))

        X_test, y_test = np.array(X_test), np.array(y_test)
        y_test = np.reshape(y_test, (y_test.shape[0],1))

    # trainDataset = Dataset(X_train, y_train, False)
    # testDataset = Dataset(X_test, y_test, False)

    return X_train, y_train, X_test, y_test, train_scaler, test_scaler