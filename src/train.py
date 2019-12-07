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
from utils import build_dataloader
from sklearn.metrics import classification_report

torch.manual_seed(0)
np.random.seed(0)

#PATH = "D:/Study/GitHub/ECG-Signal-Analysis/Data/test_1000.csv"
PATH = "D:/ecg/sample2017/sample2017/training2017"
LR = 1e-4
BATCH_SIZE = 128
EPOCH = 20
RESAMP = True

class prob_rnn(nn.Module):
    def __init__(self, input_size=29, hidden_size=100, num_layers=2, output_size=3, bidir=True):
        super(prob_rnn, self).__init__()
        self.dim = 2 if bidir == True else 1
        self.lstm1 = nn.GRU(input_size, hidden_size, num_layers, dropout=.2, batch_first=True, bidirectional=bidir)
        self.lstm2 = nn.GRU(hidden_size*self.dim, hidden_size, num_layers, dropout=.2, batch_first=True, bidirectional=bidir)
        self.linear = nn.Linear(hidden_size*self.dim, output_size)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x[:, -1, :])
        x = self.sm(x)

        return x

class cnn(nn.Module):
    def __init__(self, n_outputs=2):
        super(cnn, self).__init__()#batch*1*9000
        self.conv1 = nn.Sequential(nn.Conv1d(1, 10, 200, 10),
                                   nn.BatchNorm1d(10),
                                   nn.ReLU(),
                                   nn.Dropout(p = 0.2))#batch*10*881
        self.conv2 = nn.Sequential(nn.Conv1d(10, 10, 56, 3),
                                   nn.BatchNorm1d(10),
                                   nn.ReLU(),
                                   nn.Dropout(p = 0.2))#batch*10*276
        self.conv3 = nn.Sequential(nn.Conv1d(10, 5, 36, 3),
                                   nn.BatchNorm1d(5),
                                   nn.ReLU(), 
                                   nn.Dropout(p = 0.2))#batch*5*81
        self.conv4 = nn.Sequential(nn.Conv1d(5, 5, 21, 3),
                                   nn.BatchNorm1d(5),
                                   nn.ReLU(), 
                                   nn.Dropout(p = 0.2))#batch*5*21
        self.l1 = nn.Linear(5*21, 5*10)
        self.l2 = nn.Linear(5*10, n_outputs)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.l2(x)
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
            x = x.view(-1, 2, 29)
            out = net(x)  
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        y_pred, y_true = [], []
        for _, (x, y) in enumerate(test):
            x, y = x.view(-1, 2, 29), y.numpy()
            pred = torch.max(net(x), 1)[1].data.numpy()
            y_pred += pred.tolist()
            y_true += y.tolist()
        
        if ep%1 == 0:
            print("Epoch: ", ep, 
                  " | Loss: %.3f" % loss.data.numpy())
        report = metric(np.array(y_pred), np.array(y_true))
        loss_data.append(loss.data.numpy())
        val_score.append(0.)
        print(report)
        
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
    print("The average validation score is: %.2f" % np.mean(val_score[1:]))

def metric(y_pred, y_true):
    
    return classification_report(y_true, y_pred, target_names=['aFib', 'Normal', 'Others'])
    

if __name__ == "__main__":
    train, test = build_dataloader(PATH, RESAMP)
    net, loss, val_score = learn(train, test)
    #plot(loss, val_score)
    