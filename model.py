import pdb
import conf
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from scipy.special import softmax
from asd_model.model import Wav2Vec2ForSpeechClassification as Model
from transformers import Wav2Vec2Processor, AutoModel, AutoTokenizer, AutoProcessor, AutoConfig


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


class MMCATextModel(nn.Module):
    
    def __init__(self, n_class=1):
        super().__init__()
        
        exp_name = './asd_model/exp/checkpoint'
        self.config = AutoConfig.from_pretrained(exp_name)
        self.model = Model.from_pretrained(exp_name, config=self.config).to('cpu')
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

        
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        for param in self.model.wav2vec2.parameters():
            param.requires_grad = False
        
    
    def forward(self, x):
        wav = self.processor(x[0], sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to("cuda")

        output = self.model(wav)
        logits = output.logits.cpu().detach()


        return softmax(logits)
