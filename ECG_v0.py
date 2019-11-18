# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:55:00 2019

@author: Arcy
"""

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert
#from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)
np.random.seed(0)

PATH = "D:/Study/GitHub/ECG-Signal-Analysis/Data/test_1000.csv"
LR = 1e-2
BATCH_SIZE = 128
EPOCH = 100

'''
class ECG_dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.ecg_frame = pd.read_csv(csv_file, index_col=0)
        self.transform = transform
        le = LabelEncoder()
        self.ecg_frame.Label = le.fit_transform(self.ecg_frame.Label)

    def __len__(self):
        return len(self.ecg_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        signal = np.array(self.ecg_frame.iloc[idx, :-1].values).reshape(1,1000)
        signal = torch.from_numpy(signal).type(torch.FloatTensor)
        label = np.array(self.ecg_frame.Label.values[idx])  
        label = torch.from_numpy(label).type(torch.LongTensor)

        return (signal, label)
 '''
 
class arima():
    pass

class particle_filter():
    pass

class prob_rnn(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, num_layers=2, output_size=3, bidir=True):
        super(prob_rnn, self).__init__()
        self.dim = 2 if bidir == True else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidir)
        self.linear = nn.Linear(hidden_size*self.dim, output_size)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        x = self.sm(x)

        return x

def build_dataset(path):
    
    data = pd.read_csv(path, index_col=0)
    le, scaler = LabelEncoder(), StandardScaler()
    data.Label = le.fit_transform(data.Label)
    y = data.Label.values
    x = data.drop(["Label"], axis=1).values
    x_ = to_insfreq(x)
    scaler.fit(x_)
    x_ = scaler.transform(x_)
    x = np.hstack((x, x_))
    
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = .3, random_state=120)
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    Y_val = torch.from_numpy(Y_val).type(torch.LongTensor)
    
    train = torch.utils.data.TensorDataset(X_train, Y_train)
    val = torch.utils.data.TensorDataset(X_val, Y_val)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = False)
    
    return train_loader, val_loader

'''
def build_dataset(path):
    
    ECG_data = ECG_dataset(csv_file = path)
    dataloader = DataLoader(ECG_data, 
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    
    return dataloader, None
'''

def to_insfreq (signal, fs=300):
    
    a_signal = np.zeros((signal.shape[0], signal.shape[1]))
    instantaneous_phase = np.zeros((signal.shape[0], signal.shape[1]))
    instantaneous_frequency = np.zeros((signal.shape[0], signal.shape[1]))
    
    for i in range(signal.shape[0]):
        a_signal[i, :] = hilbert(signal[i, :])
        instantaneous_phase[i, :] = np.unwrap(np.angle(a_signal[i, :]))
        instantaneous_frequency[i, 1:] = np.diff(instantaneous_phase[i, :]) / (2.0*np.pi) * fs
    
    return instantaneous_frequency

def learn(train, test):
    
    loss_data, val_score = [], [0.]
    net = prob_rnn()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    print("Start training.")
    for ep in range(EPOCH):
        ep += 1
        for _, (x, y) in enumerate(train):
            x = x.view(-1, 2, 1000)
            out = net(x)  
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        count = 0
        for _, (x, y) in enumerate(test):
            x, y = x.view(-1, 2, 1000), y.numpy()
            pred = torch.max(net(x), 1)[1].data.numpy()
            count += (pred == y).astype(int).sum()
        accuracy = count / len(test.dataset)
        
        if ep%10 == 0:
            print("Epoch: ", ep, 
                  " | Loss: %.4f" % loss.data.numpy(), 
                  " | Val score: %.2f" % accuracy)
        loss_data.append(loss.data.numpy())
        val_score.append(accuracy)
           
    choice = input("Save or not?[y/n]")
    if choice == 'y':
        torch.save({'model_state_dict': net.state_dict()}, "net_params.pkl")
    
    return net, loss_data, val_score

def plot(loss, val_score):
    
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(loss)), loss)
    plt.title("Loss")
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(val_score)), val_score)
    plt.title("Validation accuracy")
    print("The average validation score is: ", np.mean(val_score))

if __name__ == "__main__":
    train, test = build_dataset(PATH)
    net, loss, val_score = learn(train, test)
    plot(loss, val_score)
    