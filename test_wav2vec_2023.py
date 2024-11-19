import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
from pytorch_lightning import Trainer
import sys
sys.path.append("/root/Speech2Intent/s2i-baselines")

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MMCATextModel
from dataset import S2ITEXTDataset, S2ITEXTDataset2, collate_fn2, collate_fn4
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

dataset = S2ITEXTDataset(
    csv_path="dump/raw/train_asr2cls_sp/text_dev0",
    wav_dir_path="dump/raw/train_asr2cls_sp/wav_dev0.scp",
)     

testloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=4,
        collate_fn = collate_fn2,
    )

model = MMCATextModel()
model.eval()

trues=[]
preds = []

for x, label in tqdm(testloader):
    model.to("cuda")
    probs = model((x[0].to("cuda"), x[1]))[0]
    pred = probs.round().astype(int)
    trues.append(label[0].detach().cpu().numpy().astype(int))
    preds.append(pred)
print(f"Accuracy Score = {accuracy_score(trues, preds)}\nF1-Score = {f1_score(trues, preds, average='binary', pos_label=1)}\nPrecison = {precision_score(trues, preds, pos_label=1, average='binary')}\nRecall = {recall_score(trues, preds, pos_label=1, average='binary')}")
