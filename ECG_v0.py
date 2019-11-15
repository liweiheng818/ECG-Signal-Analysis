# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:55:00 2019

@author: Arcy
"""

import torch
import torch.utils.data
import torch.nn as nn
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

PATH = "D:/Study/GitHub/ECG-Signal-Analysis/Data/test_1000.csv"
LR = 1e-2
BATCH_SIZE = 1
EPOCH = 100
TOL = 1e-3

class arima():
    pass

class particle_filter():
    pass

class prob_rnn(nn.Module):
    def __init__(self, input_size=1000, hidden_size=128, num_layers=2, output_size=3):
        super(prob_rnn, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        
        return x

def build_dataset(path):
    
    data = pd.read_csv(path)
    data.drop(["Unnamed: 0"], axis=1, inplace=True)
    le = LabelEncoder()
    data.Label = le.fit_transform(data.Label)
    y = data.Label.values
    x = data.drop(["Label"], axis=1).values
    
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

def learn(train, test):
    
    loss_data, val_score = [], []
    net = prob_rnn()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    for ep in range(EPOCH):
        ep += 1
        for _, (x, y) in enumerate(train):
            var_x, var_y = Variable(x.view(-1, BATCH_SIZE, 1000)), Variable(y.view(BATCH_SIZE))
            out = net(var_x)  
            loss = loss_func(out, var_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
        count = 0
        for _, (x, y) in enumerate(test):
            var_x, var_y = Variable(x.view(-1, BATCH_SIZE, 1000)), y.view(BATCH_SIZE).numpy()
            pred = torch.max(net(var_x), 1)[1].data.numpy()
            count += (pred == var_y).astype(int).sum()
        accuracy = count / len(test.dataset)
    
        print("Epoch: ", ep, " | Loss: %.4f" % loss.data.numpy(), " | Val score: %.2f" % accuracy)
        loss_data.append(loss.data.numpy())
        val_score.append(accuracy)
        
    choice = input("Save or not?[y/n]")
    if choice == 'y':
        torch.save({
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_func
                    }, "net_params.pkl"
        )
    
    return net, loss_data, val_score

def plot():
    pass

if __name__ == "__main__":
    train, test = build_dataset(PATH)
    net, loss, val_score = learn(train, test)
    