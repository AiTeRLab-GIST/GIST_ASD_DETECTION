import torch
from transformers import TrainingArguments

train = False
evaluate = True
feat_ext = False
infer = False

inputs = 'path'
outputs = 'asd'
batch_size = 128

db_type = 'split_zeropadding'

save_path = f'/mnt/db'
wav_path = f'/mnt/db/{db_type}'
df_names = [f'train_{db_type}.csv', f'eval_{db_type}.csv', f'infer_{db_type}.csv']

feat_ext_dir = '/mnt/db/w2v_feat_ext/'

model_name = 'facebook/wav2vec2-base-960h'
fix_feat_ext = False 
condition = 'fix-fe' if fix_feat_ext else 'tune-fe'
exp_dir = './exp/'

pooling_mode = 'mean'
is_regression = False
sampling_rate = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
