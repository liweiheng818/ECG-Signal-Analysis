# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:22:01 2019

@author: Arcy
"""
import numpy as np
from utils import get_ecg, get_rpeak, build_dataloader
from train import learn, prob_rnn
from sklearn.preprocessing import StandardScaler

PATH = "D:/ecg/sample2017/sample2017/training2017"
BATCH_SIZE = 120
FS = 300
LENGTH = 9000
NLAGS = 999
RESAMP = True

Signals, Labels = get_ecg(PATH)

R_Intervals = np.zeros((Signals.shape[0], 29))
R_val = np.zeros((Signals.shape[0], 30))
for i, record in enumerate(Signals):
    rp, rin = get_rpeak(record, FS)
    R_Intervals[i, :] = rin
    for j, val in enumerate(rp):
        R_val[i, j] = Signals[i, val]

st = StandardScaler()
R_val = st.fit_transform(R_val)

X = np.hstack((R_val[:, :-1], R_Intervals))
y = Labels

del Signals, Labels, i, record, rp, rin, R_Intervals, R_val
train, test = build_dataloader(X, y, resamp=RESAMP, batch_size=BATCH_SIZE)
net, loss, val_score = learn(train, test)
