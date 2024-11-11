import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

class egemaps_dataset(Dataset):
    def __init__(self, df, pad_len = 94):
        self.fpath = df['path']    # File paths for each sample
        self.label = df['asd']     # Labels for each sample
        self.pad_len = pad_len     # Padding length for feature sequences

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        pad_len = self.pad_len
        n_padded = 0  # Initialize padding count
        # Load the feature file; try CSV first, fall back to text format if needed
        try:
            feat = pd.read_csv(self.fpath[index], delimiter=';').values
        except:
            feat = np.loadtxt(path, delimiter =';')
        label = self.label[index]                    # Retrieve label for the current sample
        # Apply padding if the feature sequence is shorter than the specified pad length
        if pad_len != False and len(feat) < pad_len:
            n_padded = pad_len - len(feat)
            feat = self.padding(feat)                # Pad the features up to pad_len
        # Return features, label, and padding count as PyTorch tensors
        return torch.from_numpy(feat).type(torch.float32), torch.tensor(label), n_padded

    def padding(self, feat):
        pad_len_feat = np.zeros((self.pad_len, feat.shape[1]))        # Initialize padded array
        pad_len_feat[:feat.shape[0],:] = feat                         # Fill in the original features
        return pad_len_feat
