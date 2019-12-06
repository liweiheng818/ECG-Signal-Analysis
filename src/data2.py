# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:33:23 2019

@author: Arcy
"""

#
import os
import scipy.io 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import hilbert, periodogram, welch

#
FS = 300
Length = 9000

#
def build_dataset(PATH):
    
    labels = pd.read_csv(PATH + '/' + 'REFERENCE.csv', index_col=0)
    filelist = os.listdir(PATH)
    
    Signals = []
    Labels = []
    
    for file in filelist:
        f = PATH + '/' + file
        if file.endswith(".mat"):
            data = scipy.io.loadmat(f)['val']
            l = labels.loc[file.split('.')[0], 'label']
            if l == 'N' or l == 'A':
                length = data.shape[1]
                i = 0
                while length >= Length:
                    Signals.append(data[0, i:i+9000])
                    Labels.append(l)
                    i += 9000
                    length -= 9000
    
    Signals = np.array(Signals).reshape(-1, 9000)
    Labels = np.array(Labels)
    Insfreq = to_insfreq(Signals, FS)
    #Entropy = spectral_entropy(Signals, FS, normalize=True)

    
    le = LabelEncoder()
    st = StandardScaler()
    
    Labels = le.fit_transform(Labels)
    st.fit(Signals)
    Signals = st.transform(Signals)
    st.fit(Insfreq)
    Signals = st.transform(Insfreq)
    
    x = np.hstack((Insfreq, Signals))
    dictionary = {"Signals": x, "Labels": Labels}

    return dictionary

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

if __name__ == "__main__":
    PATH = "D:/ecg/sample2017/sample2017/training2017"
    dic = build_dataset(PATH)