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
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(7)

# get ecg signals from records
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

#
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
            
        segments.append(signal[l-l_padding:r+r_padding].copy())
    
    segments = np.array(segments)
    labels = np.repeat(label, segments.shape[0])
    return np.hstack((segments, labels[:, np.newaxis]))

#
def plot(loss, val_score):
    
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(loss)), loss)
    plt.title("Loss")
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(val_score)), val_score)
    plt.title("Validation accuracy")
    print("The average validation score is: %.2f" % np.mean(val_score[1:]))

#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    