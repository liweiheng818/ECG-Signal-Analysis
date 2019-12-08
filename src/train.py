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
from sklearn.metrics import classification_report, f1_score

torch.manual_seed(0)
np.random.seed(0)

class prob_rnn(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, num_layers=2, output_size=3, bidir=True):
        super(prob_rnn, self).__init__()
        self.dim = 2 if bidir == True else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, dropout=.1, batch_first=True, bidirectional=bidir)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, dropout=.1, batch_first=True, bidirectional=bidir)
        self.linear = nn.Linear(hidden_size*self.dim, output_size)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        e_output, e_hidden = self.encoder(x)
        d_output, d_hidden = self.decoder(x, e_hidden)
        out = self.linear(d_output[:, -1, :])
        out = self.sm(out)

        return out

class encoder(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, num_layers=2, bidir=True):
        super(encoder, self).__init__()
        self.dim = 2 if bidir == True else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=.1, batch_first=True, bidirectional=bidir)
        
    def forward(self, x):
        output, hidden = self.lstm(x)
    
        return output, hidden
    
class decoder(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, num_layers=2, output_size=3, bidir=True):
        super(decoder, self).__init__()
        self.dim = 2 if bidir == True else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=.1, batch_first=True, bidirectional=bidir)
        self.linear = nn.Linear(hidden_size*self.dim, output_size)
        self.sm = nn.LogSoftmax(dim=1)
        
    def forward(self, x, hidden):
        output, _ = self.lstm(x, hidden)
        output = self.linear(output[:, -1, :])
        output = self.sm(output)
        
        return output
    
class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__()
        self.dim = 2
        self.input_size = [384, 1000]
        self.hidden_size = 100
        self.output_size = 3
        self.layers = 2
        self.embedding = cnn_1d(self.input_size[0])
        self.encoder = encoder(self.input_size[0], self.hidden_size, self.layers)
        self.decoder = decoder(self.input_size[0], self.hidden_size, self.layers, self.output_size)
    
    def forward(self, x):
        feature = self.embedding(x)
        e_output, e_hidden = self.encoder(feature)
        d_output = self.decoder(feature, e_hidden)

        return d_output

class cnn_1d(nn.Module):
    def __init__(self, n_outputs=3):
        super(cnn_1d, self).__init__()#batch*1*1000
        self.output = n_outputs
        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 50, 5),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.Dropout(p = 0.1))#batch*16*95
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, 45, 5),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.Dropout(p = 0.1))#batch*64*5
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(), 
                                   nn.Dropout(p = 0.1))#batch*128*3
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)

        return x.view(-1, 1, self.output)

def learn(net, train, test, lr=1e-4, epoch=50):
    
    loss_data, val_score = [], [0.]
    network = net.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    print("Start training.")
    for ep in range(epoch):
        ep += 1
        for _, (x, y) in enumerate(train):
            x = x.view(-1, 1, 1000).cuda()
            out = network(x).cuda()
            loss = loss_func(out, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        y_pred, y_true = [], []
        for _, (x, y) in enumerate(test):
            x, y = x.view(-1, 1, 1000).cuda(), y.numpy()
            pred = torch.max(network(x), 1)[1].cuda().data
            pred = pred.cpu().numpy()
            y_pred += pred.tolist()
            y_true += y.tolist()
        
        score = f1_score(y_true, y_pred, average='micro')
        if ep%1 == 0:
            print("Epoch: ", ep, 
                  " | Loss: %.3f" % loss.data.cpu().numpy(),
                  " | F1 Score: %.2f" % score)
            print(metric(np.array(y_pred), np.array(y_true))) 
            
        loss_data.append(loss.data.cpu().numpy())
        val_score.append(score)
        
    choice = input("Save or not?[y/n]")
    if choice == 'y':
        torch.save({'model_state_dict': net.state_dict()}, "net_params.pkl")
    
    return loss_data, val_score

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
    