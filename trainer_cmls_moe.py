from model import CMMOEModel2, CMMOEModel
from dataset import S2IMELTEXTDataset
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from loss import Binary_MutualLoss, InfoNCELoss


# SEED
SEED=2024
pl.utilities.seed.seed_everything(SEED)
torch.manual_seed(SEED)

# update the wandb online/offline model and CUDA device
import os
os.environ['WANDB_MODE'] = 'online'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.optim.lr_scheduler import LambdaLR


class LightningModel(pl.LightningModule):
    def __init__(self, bert_name='kykim/bert-kor-base'):
        super().__init__()
        # tiny/small model
        print(bert_name)
        # self.model = CMMOEModel(bert_name)
        self.model = CMMOEModel2(bert_name)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return [optimizer]

    def loss_fn(self, prediction, targets):
        return nn.BCEWithLogitsLoss()(prediction, targets)
    
    def cl_loss_fn(self, prediction, targets):
        return InfoNCELoss(temperature=0.1)(prediction, targets)
    
    def mutual_loss_fn(self, prediction, targets):
        return Binary_MutualLoss()(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        asr_output, nlp_output, asr_embed, nlp_embed, loss_reg = self(x)

        asr_loss = self.loss_fn(asr_output, y.reshape(-1,1).float())
        nlp_loss = self.loss_fn(nlp_output, y.reshape(-1,1).float())
        mutual_loss = self.mutual_loss_fn(asr_output, nlp_output)
        l2_loss = self.cl_loss_fn(asr_embed, nlp_embed)

        batch_weight = F.softmax(torch.randn(4), dim=-1).to(self.device)
        
        train_losses = torch.zeros(4).to(self.device)
        
        train_losses[0] = asr_loss
        # train_losses[3] = loss
        train_losses[1] = nlp_loss
        train_losses[2] = mutual_loss
        train_losses[3] = l2_loss
        
        # total_loss = 0.7*asr_loss + 0.3*loss + 0.2*l2_loss + 0.5*nlp_loss
        total_loss = torch.mul(train_losses, batch_weight).sum()

        acc = (self.sigmoid(asr_output).squeeze().round() == y).float().mean()

        self.log('train/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'loss':total_loss, 
            'acc':acc
            }


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        asr_output, nlp_output, asr_embed, nlp_embed, loss_reg = self(x)

        asr_loss = self.loss_fn(asr_output, y.reshape(-1,1).float())
        nlp_loss = self.loss_fn(nlp_output, y.reshape(-1,1).float())
        mutual_loss = self.mutual_loss_fn(asr_output, nlp_output)
        l2_loss = self.cl_loss_fn(asr_embed, nlp_embed)

        batch_weight = F.softmax(torch.randn(4), dim=-1).to(self.device)
        
        train_losses = torch.zeros(4).to(self.device)
        
        train_losses[0] = asr_loss
        # train_losses[3] = loss
        train_losses[1] = nlp_loss
        train_losses[2] = mutual_loss
        train_losses[3] = l2_loss
        
        # total_loss = 0.7*asr_loss + 0.3*loss + 0.2*l2_loss + 0.5*nlp_loss
        total_loss = torch.mul(train_losses, batch_weight).sum()

        output = self.sigmoid(asr_output).squeeze().round()
        acc = (output == y).float().mean()
        output = output.detach().cpu().numpy().astype(int)
        y = y.detach().cpu().numpy().astype(int)

        
        TP = (output & y).sum()
        FP = (output & ~y).sum()
        FN = (~output & y).sum()
        # Calculate precision, recall, and F1 score
        if (TP+FP) <= 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        if (TP+FN) <= 0:
            recall = 0.0
        else:
            recall = TP / (TP + FN)
        if (precision + recall) <= 0:
            f1_score=0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        self.log('val/asr_loss' , asr_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/total_loss' , total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision',precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall',recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1_score',f1_score, on_step=False, on_epoch=True, prog_bar=True)
        

        return {'val_loss':total_loss, 
                'val_acc':acc,
                'val_precision':precision,
                'val_recall':recall,
                'val_f1_score':f1_score,
                }
    

        
    

def main(args):
    k_fold = 5
    for i in range(k_fold):
        # skit-s2i dataset
        train_dataset = S2IMELTEXTDataset(
            csv_path="dump/raw/train_da_sum_sp/text_train"+str(i),
            wav_dir_path="dump/raw/train_da_sum_sp/wav_train"+str(i)+".scp",
        )
        val_dataset = S2IMELTEXTDataset(
            csv_path="dump/raw/train_da_sum_sp/text_dev"+str(i),
            wav_dir_path="dump/raw/train_da_sum_sp/wav_dev"+str(i)+".scp",
        )   

        # dataloaders
        trainloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=8, 
                shuffle=True, 
                num_workers=8,
            )
        
        valloader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=8, 
                num_workers=8,
            )

        model = LightningModel(args.bert_name)
        name = args.bert_name.split('/')[1].split('-')[0]
        # update the logger to Wandb or Tensorboard
        run_name = f"da-cmls-moe-whisper-nolstm-last-{name}_{i}"
        logger = WandbLogger(
            name=run_name,
            project='S2I-baseline'+str(i)
        )

        model_checkpoint_callback = ModelCheckpoint(
                dirpath='checkpoints',
                monitor='val/f1_score', 
                mode='max',
                verbose=1,
                filename=run_name + "-epoch={epoch}.ckpt")

        early_stop_callback = EarlyStopping(
                monitor='val/f1_score',  # 모니터링할 메트릭
                patience=5,          # 개선이 없을 때 기다릴 에포크 수
                verbose=True,        # 로깅 여부
                mode="max"           # 메트릭 최소화 (val_loss를 최소화)
        )

        trainer = Trainer(
                # accumulate_grad_batches=32, 
                # gradient_clip_val=1.0,
                fast_dev_run=False, # true for dev run
                gpus=1, 
                max_epochs=12, 
                callbacks=[
                    model_checkpoint_callback,
                    early_stop_callback
                ],
                logger=logger,
                )

        trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MTL model")

    # 명령줄 인자 추가
    parser.add_argument('--bert_name', type=str, default='kykim/bert-kor-base', help="Model name to use in MTLModel")

    args = parser.parse_args()
    main(args)
