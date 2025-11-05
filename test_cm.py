import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


import sys
sys.path.append("/root/Speech2Intent/s2i-baselines")

import torch
import torch.nn as nn
import torch.nn.functional as F

# choose the model
# from trainer_cmls import LightningModel
from trainer_cmls_moe_self import LightningModel
# from trainer_cmls_conformer_moe import LightningModel

from dataset import S2IMELTEXTDataset, collate_fn2

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

dataset = S2IMELTEXTDataset(
        csv_path="dump/raw/test_da/text",
        wav_dir_path="dump/raw/test_da/wav.scp",
    )

testloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=4,
        num_workers=4,
    )

# change path to checkpoint

model = LightningModel.load_from_checkpoint("checkpoints/da-cmls-moe-whisper-self-cl_07-bert_4-epoch=epoch=5.ckpt.ckpt")
model.to('cuda')
model.eval()

trues=[]
preds = []

for x, label in tqdm(testloader):

    y_hat_l, _, _, _, _ = model((x[0].to("cuda"), x[1]))
    # _, _, y_hat_l = model((x_tensor, x[1]))

    probs = F.sigmoid(y_hat_l).squeeze().round()
    pred = probs.detach().cpu().numpy()
    trues.extend(label)
    preds.extend(pred)
print(f"Accuracy Score = {accuracy_score(trues, preds)}\nF1-Score = {f1_score(trues, preds, average='binary', pos_label=1)}\nPrecison = {precision_score(trues, preds, pos_label=1, average='binary')}\nRecall = {recall_score(trues, preds, pos_label=1, average='binary')}")