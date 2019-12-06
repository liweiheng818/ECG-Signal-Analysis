# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:55:00 2019

@author: Arcy
"""

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data import build_dataloader

torch.manual_seed(0)
np.random.seed(0)

#PATH = "D:/Study/GitHub/ECG-Signal-Analysis/Data/test_1000.csv"
PATH = "D:/ecg/sample2017/sample2017/training2017"
LR = 1e-2
BATCH_SIZE = 128
EPOCH = 20

class arima():
    pass

class prob_rnn(nn.Module):
    def __init__(self, input_size=9000, hidden_size=100, num_layers=2, output_size=2, bidir=True):
        super(prob_rnn, self).__init__()
        self.dim = 2 if bidir == True else 1
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidir)
        self.linear = nn.Linear(hidden_size*self.dim, output_size)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        x = self.sm(x)

        return x

def learn(train, test):
    
    loss_data, val_score = [], [0.]
    net = prob_rnn()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    print("Start training.")
    for ep in range(EPOCH):
        ep += 1
        for _, (x, y) in enumerate(train):
            x = x.view(-1, 2, 9000)
            out = net(x)  
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        count = 0
        for _, (x, y) in enumerate(test):
            x, y = x.view(-1, 2, 9000), y.numpy()
            pred = torch.max(net(x), 1)[1].data.numpy()
            count += (pred == y).astype(int).sum()
        accuracy = count / len(test.dataset)
        
        if ep%1 == 0:
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
    train, test = build_dataloader(PATH)
    net, loss, val_score = learn(train, test)
    plot(loss, val_score)
    