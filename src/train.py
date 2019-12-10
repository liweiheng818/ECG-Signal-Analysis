# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:55:00 2019

@author: Arcy
"""

#
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

from sklearn.metrics import classification_report, f1_score

torch.manual_seed(0)
np.random.seed(0)

# classifier
#
class bi_lstm(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, num_layers=2, output_size=3, bidir=True):
        
        super(bi_lstm, self).__init__()
        self.dim = 2 if bidir == True else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm1 = nn.LSTM(self.input_size, 
                             self.hidden_size, 
                             num_layers, 
                             dropout=.2, 
                             batch_first=True, bidirectional=bidir)
        self.linear = nn.Linear(self.hidden_size*self.dim, self.output_size)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x = self.linear(x[:, -1, :])
        x = self.sm(x)

        return x

# feature extractor
#
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
        self.conv4 = nn.Sequential(nn.Conv1d(128, 256, 3, 1),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(), 
                                   nn.Dropout(p = 0.1))#batch*256*1
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x.view(-1, 1, self.output)

#
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

# model
#
class cnn_feed_lstm(nn.Module):# cnn -> lstm -> out
    def __init__(self, ):
        
        super(cnn_feed_lstm, self).__init__()
        self.dim = 2
        self.input_size = 256
        self.hidden_size = 100
        self.output_size = 3
        self.layers = 2
        self.cnn = cnn_1d(self.input_size)
        self.lstm = bi_lstm(self.input_size, self.hidden_size, self.layers, self.output_size)
    
    def forward(self, x):
        
        feature = self.cnn(x)
        out = self.lstm(feature)

        return out

class cnn_concat_lstm(nn.Module):
    def __init__(self, ):
        
        super(cnn_concat_lstm, self).__init__()
    
    def forward(self, x):
        
        pass

#
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
        torch.save({'model_state_dict': net.state_dict()}, "../params/net_params.pkl")
    
    return np.array(loss_data), np.array(val_score)

#
def metric(y_pred, y_true):
    
    return classification_report(y_true, y_pred, target_names=['aFib', 'Normal', 'Others'])
    