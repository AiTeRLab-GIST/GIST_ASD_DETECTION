import pdb
import conf
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

class lstm_block(nn.Module):
    def __init__(self, in_shape, hidden_size, batch_first):
        super().__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(in_shape, hidden_size, num_layers = 1, batch_first = batch_first, bidirectional = True)
        self.BN = nn.BatchNorm1d(2 * hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.BN(x)
        return x

class dense_block(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.linear = nn.Linear(in_shape, out_shape)
        self.BN = nn.BatchNorm1d(out_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if len(x.shape) == 3:
            x = self.BN(x.permute(0,2,1)).permute(0,2,1)
        else:
            x = self.BN(x)
        x = self.relu(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = dense_block(88, 70)
        self.dense2 = dense_block(70, 54)
        self.dense3 = dense_block(54, 70)
        self.proj = nn.Linear(70,88)

    def forward(self, x):
        x = self.dense1(x)
        z = self.dense2(x)
        x = self.dense3(z)
        x = self.proj(x)
        return x, z

class MultiTaskAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = dense_block(88, 70)
        self.dense2 = dense_block(70, 54)
        self.dense3 = dense_block(54, 70)
        self.proj = nn.Linear(70,88)
        self.aux = nn.Linear(54,2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.dense1(x)
        z = self.dense2(x)
        x = self.dense3(z)
        x = self.proj(x)
        aux = self.aux(z)
        logits = self.softmax(aux)
        return x, z, logits

class BLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.aux = nn.Linear(54, 128)
        self.lstm = lstm_block(128, 128, True)
        self.dense1a = dense_block(256, 128)
        self.dense2a = dense_block(128, 64)
        self.dense3a = dense_block(64, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.aux(x)
        x = self.lstm(x)
        x = self.dense1a(x)
        x = self.dense2a(x)
        x = self.dense3a(x)
        return self.softmax(x) 

class AEBLSTMJT(nn.Module):
    def __init__(self):
        super().__init__()
        condition = conf.condition
        self.AEPart = AutoEncoder()
        self.BLSTM_clsf = BLSTMClassifier()

    def forward(self, x, feat_ext=False):
        recons, h = self.AEPart(x)
        logits = self.BLSTM_clsf(h)
        return recons, h, logits

class AEBLSTMFT(nn.Module):
    def __init__(self):
        super().__init__()
        condition = conf.condition
        self.AEPart = MultiTaskAutoEncoder()
        for param in self.AEPart.parameters():
            param.requires_grad = False
        self.BLSTM_clsf = BLSTMClassifier()

    def forward(self, x, feat_ext=False):
        recons, h, _ = self.AEPart(x)
        logits = self.BLSTM_clsf(h)
        return recons, h, logits
