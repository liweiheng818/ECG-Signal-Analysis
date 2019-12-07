# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:33:23 2019

@author: Arcy
"""

#
import os
import torch
import scipy.io 
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import hilbert, periodogram, welch
from statsmodels.tsa.stattools import acf
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal
#
torch.manual_seed(7)
np.random.seed(7)

BATCH_SIZE = 120
FS = 300
LENGTH = 9000
NLAGS = 999

#
def build_dataloader(X, y, resamp=True, batch_size=120):
    
    df = pd.DataFrame(data=np.hstack((X, y)), 
                      columns=[i for i in range(X.shape[1])] + ['Class'])
    
    y = df.Class
    X = df.drop('Class', axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.15, random_state=77)
    
    if resamp == True:
        X_train, y_train = resample_data(X_train, y_train)
        X_val, y_val = resample_data(X_val, y_val)
    
    X_train = X_train.values
    X_val = X_val.values
    y_train = y_train.values.astype('int')
    y_val = y_val.values.astype('int')
    
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.LongTensor)
    
    train = torch.utils.data.TensorDataset(X_train, y_train)
    val = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
    
    return train_loader, val_loader

def get_ecg(path):
    
    labels = pd.read_csv(path + '/' + 'REFERENCE.csv', index_col=0)
    filelist = os.listdir(path)
    
    Signals = []
    Labels = []
    
    for file in filelist:
        f = path + '/' + file
        if file.endswith(".mat"):
            data = scipy.io.loadmat(f)['val']
            l = labels.loc[file.split('.')[0], 'label']
            if l != '~':
                length = data.shape[1]
                i = 0
                while length >= LENGTH:
                    Signals.append(data[0, i:i+LENGTH])
                    Labels.append(l)
                    i += LENGTH
                    length -= LENGTH
    
    Signals = np.array(Signals).reshape(-1, LENGTH)
    Labels = np.array(Labels)
    
    le = LabelEncoder()
    st = StandardScaler()
    
    Labels = le.fit_transform(Labels)
    Labels = Labels[:, np.newaxis]
    Signals = st.fit_transform(Signals)
    
    return Signals, Labels

def get_rpeak(signal, fs=300):

    order = int(0.3 * fs)
    signal, _, _ = filter_signal(signal, 
                                 ftype='FIR', 
                                 band='bandpass', 
                                 order=order, 
                                 frequency=[3, 45], 
                                 sampling_rate=fs)
    rpeaks_h, = ecg.hamilton_segmenter(signal=signal, sampling_rate=fs)
    rpeaks_h = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks_h, sampling_rate=fs, tol=0.05)
    rpeaks_h = np.array(rpeaks_h)
    rpeaks_h = rpeaks_h.reshape((rpeaks_h.shape[0]*rpeaks_h.shape[1]))
    
    #hbs, _ = ecg.extract_heartbeats(signal=signal, 
    #                                rpeaks=rpeaks_h, 
    #                               sampling_rate=fs, 
    #                                before=0.2, 
    #                                after=0.4)
    
    rr = np.zeros((30)).astype('int')
    
    n = rpeaks_h.shape[0]
    if  n <= 30:
        rr[:n] = rpeaks_h
    else:
        rr = rpeaks_h[:30]
        
    rr_in = np.diff(rr)
    
    return rr, rr_in



def to_insfreq (signal, fs=300):
    
    instantaneous_frequency = np.zeros((signal.shape[0], signal.shape[1]))
    
    for i in range(signal.shape[0]):
        a_signal = hilbert(signal[i, :])
        instantaneous_phase = np.unwrap(np.angle(a_signal))
        instantaneous_frequency[i, 1:] = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
    
    return instantaneous_frequency

def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False):
     
    se = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if method == 'fft':
                _, psd = periodogram(x, sf)
            elif method == 'welch':
                _, psd = welch(x, sf, nperseg=nperseg)
            psd_norm = np.divide(psd, psd.sum())
            se[i, j] = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
            if normalize:
                se[i, j] /= np.log2(psd_norm.size)
    
    return se

def to_acf(signal, nlags):
    
    res = np.zeros((signal.shape[0], NLAGS+1))
    for i in range(NLAGS):
        res[i, :] = acf(signal[i, :], nlags=nlags)
    
    return res 

def resample_data(data, label):
    
    X = pd.concat([data, label], axis=1)
        
    aFib = X[X.Class==0]
    normal = X[X.Class==1]
    other = X[X.Class==2]
        
    aFib_new = resample(aFib,
                        replace=True, 
                        n_samples=len(normal),
                        random_state=77)
    
    other_new = resample(other,
                         replace=True,
                         n_samples=len(normal),
                         random_state=77)
    
    df_n = pd.concat([normal, aFib_new, other_new])
    y_ = df_n.Class
    X_ = df_n.drop('Class', axis=1)
    
    return X_, y_

if __name__ == "__main__":
    PATH = "D:/ecg/sample2017/sample2017/training2017"
    Signals, Labels = get_ecg(PATH)
    Rp = np.zeros((Signals.shape[0], 30))
    count = 1000
    for i, record in enumerate(Signals):
        rp = get_rpeak(record, FS)
        if rp.shape[0] < count:
            count = rp.shape[0]
        Rp[i, :] = rp
    