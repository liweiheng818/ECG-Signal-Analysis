# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:26:29 2019

@author: Arcy
"""

import torch
#from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from data2 import build_dataset
#from torch.utils.data import Dataset, DataLoader

#torch.manual_seed(0)

BATCH_SIZE = 120

'''
class ECG_dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.ecg_frame = pd.read_csv(csv_file, index_col=0)
        self.transform = transform
        le = LabelEncoder()
        self.ecg_frame.Label = le.fit_transform(self.ecg_frame.Label)

    def __len__(self):
        return len(self.ecg_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        signal = np.array(self.ecg_frame.iloc[idx, :-1].values).reshape(1,1000)
        signal = torch.from_numpy(signal).type(torch.FloatTensor)
        label = np.array(self.ecg_frame.Label.values[idx])  
        label = torch.from_numpy(label).type(torch.LongTensor)

        return (signal, label)

def build_dataset(path):
    
    ECG_data = ECG_dataset(csv_file = path)
    dataloader = DataLoader(ECG_data, 
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    
    return dataloader, None
'''

def build_dataloader(path):
    '''
    data = pd.read_csv(path, index_col=0)
    le = LabelEncoder()
    data.Label = le.fit_transform(data.Label)
    y = data.Label.values
    x = data.drop(["Label"], axis=1).values
    
    scaler = StandardScaler()
    x_ = to_insfreq(x, FS)
    scaler.fit(x_)
    x_ = scaler.transform(x_)
    x = x_
    
    se = spectral_entropy(x, FS, normalize=True)
    scaler.fit(x_)
    x_ = scaler.transform(x_)
    print(se.shape)
    x__ = np.hstack((se, x_))
    '''
    
    data = build_dataset(path)
    x = data['Signals']
    y = data['Labels']
    
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = .3, random_state=120)
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    Y_val = torch.from_numpy(Y_val).type(torch.LongTensor)
    
    train = torch.utils.data.TensorDataset(X_train, Y_train)
    val = torch.utils.data.TensorDataset(X_val, Y_val)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = False)
    
    return train_loader, val_loader
