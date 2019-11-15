# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:55:00 2019

@author: Arcy
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from torch.autograd import Variable

LR = 1e-2
BATCH_SIZE = 16
EPOCH = 100
TOL = 1.

class arima():
    pass

class particle_filter():
    pass

class prob_rnn(nn.Module):
    def __init__(self, input_size=1000, hidden_size=40, num_layers=2, output_size=3):
        super(prob_rnn, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.sm = nn.Softmax(output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        output = self.sm(x)
        
        return output

def build_dataset():
    pass

def learn(train):
    net = prob_rnn()
    ep = 0
    loss = 100
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    while ep < EPOCH and loss > TOL:
        for _, (x, y) in enumerate(train):
            var_x, var_y = Variable(x), Variable(y)
            out = net(var_x)
            loss = loss_func(out, var_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    choice = input("Save or not?[y/n]")
    if choice == 'y':
        torch.save({
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_func
                    }, "net_params.pkl"
        )
    
    return net

def valid(test):
    pass

if __name__ == "__main__":
    train, test = build_dataset()
    net = learn(train)
    valid(net, test)
    
    