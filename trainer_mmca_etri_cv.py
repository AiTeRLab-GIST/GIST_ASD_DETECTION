from model import MMCATextModel2, MMCATextModel3
from dataset import S2ITEXTDataset2, S2ITEXTDataset, collate_fn2, collate_fn4

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
import lightning_lite


# SEED
SEED=2024
lightning_lite.utilities.seed.seed_everything(SEED)
torch.manual_seed(SEED)

# update the wandb online/offline model and CUDA device
import os
os.environ['WANDB_MODE'] = 'online'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from torch.optim.lr_scheduler import LambdaLR


class LightningModel(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        # tiny/small model
        self.model = MMCATextModel2() 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return [optimizer]


    def loss_fn(self, prediction, targets):
        return nn.BCEWithLogitsLoss()(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        asr_output = self(x)

        asd_loss = self.loss_fn(asr_output, y.reshape(-1,1).float())
        

        total_loss = asd_loss
        
        acc = (self.sigmoid(asr_output).squeeze().round() == y).float().mean()

        self.log('train/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)

        return {
            'loss':total_loss, 
            'acc':acc
            }


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        asr_output = self(x)


        asd_loss = self.loss_fn(asr_output, y.reshape(-1,1).float())
        
        total_loss = asd_loss

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

        self.log('val/total_loss' , total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)
        self.log('val/acc',acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)
        self.log('val/precision',precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)
        self.log('val/recall',recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)
        self.log('val/f1_score',f1_score, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)
        

        return {
                'val_acc':acc,
                'val_precision':precision,
                'val_recall':recall,
                'val_f1_score':f1_score,
                }
    

        
    

if __name__ == "__main__":
    k_fold = 5
    for i in range(k_fold):
        # skit-s2i dataset
        train_dataset = S2ITEXTDataset(
            csv_path="dump/raw/train_da_sum_sp/text_train"+str(i),
            wav_dir_path="dump/raw/train_da_sum_sp/wav_train"+str(i)+".scp",
        )
        val_dataset = S2ITEXTDataset(
            csv_path="dump/raw/train_da_sum_sp/text_dev"+str(i),
            wav_dir_path="dump/raw/train_da_sum_sp/wav_dev"+str(i)+".scp",
        ) 
        # train_dataset = S2ITEXTDataset(
        #     csv_path="dump/raw/train_asr2cls_sp/text_train"+str(i),
        #     wav_dir_path="dump/raw/train_asr2cls_sp/wav_train"+str(i)+".scp",
        # )
   

        # dataloaders
        trainloader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=4, 
                shuffle=True,
                collate_fn = collate_fn2, 
                num_workers=4,
            )
        
        valloader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=4,
                collate_fn = collate_fn2,
                num_workers=4,
            )

        model = LightningModel()

        # update the logger to Wandb or Tensorboard
        run_name = "cls-mmca-text-wav2vec5_"+str(i)
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

        trainer = Trainer(
                # accumulate_grad_batches=32, 
                # gradient_clip_val=1.0,
                fast_dev_run=False, # true for dev run
                gpus=1, 
                max_epochs=12, 
                callbacks=[
                    model_checkpoint_callback,
                ],
                logger=logger,
                )

        trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
        
