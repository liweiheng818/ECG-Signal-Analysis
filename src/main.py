# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:22:01 2019

@author: Arcy
"""

#
import torch
import numpy as np

from utils import get_ecg, qrs_detection, get_segments, plot
from data import build_dataloader
from train import learn, cnn_feed_lstm

#
PATH = "D:/ecg/sample2017/sample2017/training2017"
BATCH_SIZE = 2048
EPOCH = 50
FS = 300
LENGTH = 9000
LR = 1e-3
RESAMP = False

#
try:
    segments = np.load('../data/segment.npy')
except:
    signals, labels = get_ecg(PATH, length=LENGTH)
    segments = np.zeros((245990, 1001))
    k = 0
    
    for i, record in enumerate(signals):
        rp = qrs_detection(record, sample_rate=FS)
        seg = get_segments(record, rp, labels[i])
        if seg is not None:
            segments[k:k+seg.shape[0], :] = seg
            k += seg.shape[0]
    del signals, labels
    
    np.save('./data/segment.npy', segments)

X, y = segments[:, :-1], segments[:, -1][:, np.newaxis]
del segments

train, test = build_dataloader(X, y, resamp=RESAMP, batch_size=BATCH_SIZE)
del X, y

net = cnn_feed_lstm()
try:
    params = torch.load("../params/net_0.81.pkl")
    net.load_state_dict(params["model_state_dict"])
except:
    pass

loss, val_score = learn(net, train, test, lr=LR, epoch=EPOCH)
plot(loss, val_score)

