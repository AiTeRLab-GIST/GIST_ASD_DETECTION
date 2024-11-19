import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import torchaudio
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset

class egemaps_dataset(Dataset):
    def __init__(self, df, pad_len = 94):
        self.fpath = df['path']
        self.label = df['asd'] 
        self.pad_len = pad_len

    def __len__(self):
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


def collate_fn2(batch):
    (seq, label) = zip(*batch)
    (wav, text) = zip(*seq)
    seql = [x[0].reshape(-1,) for x in wav]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    label = torch.tensor(list(label))
    return (data, list(text)), label

def collate_fn4(batch):
    (seq, label) = zip(*batch)
    (wav, cls_text, text) = zip(*seq)
    seql = [x[0].reshape(-1,) for x in wav]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    label = torch.tensor(list(label))
    return (data, list(cls_text), list(text)), label


def pad_or_truncate_list(input_list, desired_length=1500, pad_value=0):
    # 리스트의 현재 길이
    current_length = len(input_list)
    
    if current_length < desired_length:
        # 리스트가 주어진 길이보다 짧은 경우, 패딩 추가
        return input_list + [pad_value] * (desired_length - current_length)
    else:
        # 리스트가 주어진 길이 이상인 경우, 주어진 길이까지 잘라냄
        return input_list[:desired_length]


class S2ITEXTDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path=None, wav_dir_path=None):
        self.df1 = pd.read_csv(csv_path, sep='\t', encoding='utf-8', header=None)
        self.wav_df = pd.read_csv(wav_dir_path, sep=' ', encoding='utf-8', header=None)
        self.wav_df = self.wav_df.iloc[:, 1:]
        self.df = pd.concat([self.df1, self.wav_df], axis=1)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label_to_int = {'ASD': 1, 'TD': 0, 'OD': 2}
        row = self.df.iloc[idx]
        intent_class = row[0].split(' ')[-1]
        text = ' '.join(row[0].split(' ')[1:-1])
        wav_path = row[1]
        # wav_path = row[1]

        wav_tensor, _= torchaudio.load(wav_path)

        intent_class = int(label_to_int[intent_class])
        return (wav_tensor, text), intent_class
        
