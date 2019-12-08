# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:33:23 2019

@author: Arcy
"""

#
import os
import scipy.io 
import pywt as pw
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(7)

#
def get_ecg(path, length=9000):
    
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
                while length >= length:
                    Signals.append(data[0, i:i+length])
                    Labels.append(l)
                    i += length
                    length -= length
    
    Signals = np.array(Signals).reshape(-1, length)
    Labels = np.array(Labels)
    
    le = LabelEncoder()
    st = StandardScaler()
    
    Labels = le.fit_transform(Labels)
    Labels = Labels[:, np.newaxis]
    Signals = st.fit_transform(Signals)
    
    return Signals, Labels

#
def qrs_detection(signal, sample_rate=300, max_bpm=300):

    ## Stationary Wavelet Transform
    coeffs = pw.swt(signal, wavelet = "haar", level=2, start_level=0, axis=-1)
    d2 = coeffs[1][1] ##2nd level detail coefficients
    
    
    ## Threhold the detail coefficients
    avg = np.mean(d2)
    std = np.std(d2)
    sig_thres = [abs(i) if abs(i)>2.0*std else 0 for i in d2-avg]
    
    ## Find the maximum modulus in each window
    window = int((60.0/max_bpm)*sample_rate)
    sig_len = len(signal)
    n_windows = int(sig_len/window)
    modulus,qrs = [],[]
    
    ##Loop through windows and find max modulus
    for i in range(n_windows):
        start = i*window
        end = min([(i+1)*window,sig_len])
        mx = max(sig_thres[start:end])
        if mx>0:
            modulus.append( (start + np.argmax(sig_thres[start:end]),mx))
    
    
    ## Merge if within max bpm
    merge_width = int((0.2)*sample_rate)
    i=0
    while i < len(modulus)-1:
        ann = modulus[i][0]
        if modulus[i+1][0]-modulus[i][0] < merge_width:
            if modulus[i+1][1]>modulus[i][1]: # Take larger modulus
                ann = modulus[i+1][0]
            i+=1
                
        qrs.append(ann)
        i+=1 
    ## Pin point exact qrs peak
    window_check = int(sample_rate/6)
    #signal_normed = np.absolute((signal-np.mean(signal))/(max(signal)-min(signal)))
    r_peaks = [0]*len(qrs)
    
    for i,loc in enumerate(qrs):
        start = max(0,loc-window_check)
        end = min(sig_len,loc+window_check)
        wdw = np.absolute(signal[start:end] - np.mean(signal[start:end]))
        pk = np.argmax(wdw)
        r_peaks[i] = start+pk
    r_peaks = np.array(r_peaks)
    
    return r_peaks

def get_segments(signal, rpeaks, label, length=1000):
    
    n = rpeaks.shape[0]
    if n <= 8:
        return 
    
    segments = []
    
    for i in range(2, n-6):
        l, r = rpeaks[i], rpeaks[i+3]
        padding = length - r + l
        if padding%2 == 0:
            l_padding = r_padding = int(padding/2)
        else:
            l_padding = int((padding - 1)/2)
            r_padding = int((padding + 1)/2)
        
        if l_padding > l:
            r_padding += l_padding - l
            l_padding = l
            
        if r + r_padding >= signal.shape[0]:
            r_padding = signal.shape[0] - 1 - r
            l_padding = l - signal.shape[0] + 1 + length
            
        #print(l-l_padding, r+r_padding)
        segments.append(signal[l-l_padding:r+r_padding].copy())
    
    segments = np.array(segments)
    labels = np.repeat(label, segments.shape[0])
    return np.hstack((segments, labels[:, np.newaxis]))

#
if __name__ == "__main__":
    
    PATH = "D:/ecg/sample2017/sample2017/training2017"
    Signals, Labels = get_ecg(PATH)
    segments = np.zeros((245990, 1001))
    k = 0
    
    for i, record in enumerate(Signals):
        rp = qrs_detection(record)
        seg = get_segments(record, rp, Labels[i])
        if seg is not None:
            segments[k:k+seg.shape[0], :] = seg
            k += seg.shape[0]
    
    del Signals, Labels
    