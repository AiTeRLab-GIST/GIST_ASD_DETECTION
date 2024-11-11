import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

class egemaps_dataset(Dataset):
    # Initialize the dataset with file paths and labels
    def __init__(self, df, pad_len = 94):
        self.fpath = df['path']
        self.label = df['asd'] 
        self.pad_len = pad_len

    def __len__(self):
        # Returns the total number of samples in the dataset.
        return len(self.label)

    def __getitem__(self, index):
        pad_len = self.pad_len
        try:
            feat = pd.read_csv(self.fpath[index], delimiter=';').values
        except:
            feat = np.loadtxt(path, delimiter =';')
        label = self.label[index]
        if pad_len != False and len(feat) < pad_len:
            n_padded = pad_len - len(feat)
            feat = self.padding(feat)
        return torch.from_numpy(feat).type(torch.float32), torch.tensor(label), n_padded

    def padding(self, feat):
        pad_len_feat = np.zeros((self.pad_len, feat.shape[1]))
        pad_len_feat[:feat.shape[0],:] = feat
        return pad_len_feat
