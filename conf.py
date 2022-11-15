import os
from datetime import datetime as dt
import torch
from transformers import TrainingArguments

inputs = 'path'
outputs = 'asd'
batch_size = 128

save_path = './data/'
wav_path = './data/egemaps/'
db_path = './data/'
df_names = ['train.csv', 'valid.csv', 'test.csv']

feat_ext_dir = '../feat_ext/ae'

condition = 'JT'

now = dt.now()
exp_dir = './exp/%02d%02d%02d'%(now.year%100, now.month, now.day)

is_regression = False
sampling_rate = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
