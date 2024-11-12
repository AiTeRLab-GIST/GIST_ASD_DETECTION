from datetime import datetime as dt
from tqdm import tqdm
import conf
import pdb
import torch
from torch import nn
import traceback
import numpy as np
from typing import Any, Dict, Union
from packaging import version
'''
from torch.optim.lr_scheduler import _LRScheduler as LRModule

class ExponentialLRwithPatience(LRModule):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]
'''

now = dt.now()

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, target_model, pad_len):
        self.model = model
        self.n_epoches = 500
        self.rgrs_criterion = nn.MSELoss()
        self.clsf_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'sum')
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.9, patience = 20)
        #torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.99)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.target_model = target_model
        self.pad_len = pad_len

    def masking_padded(self, outputs, n_padded):
        masked_outputs = []
        masks = torch.zeros(outputs.shape)
        for idx in range(len(n_padded)):
            masks[idx,self.pad_len-n_padded[idx]:,:] = 1
        masks = masks.type(torch.bool).to(conf.device)
        for Output, Pad, mask in zip(outputs, n_padded, masks):
            Output = torch.masked_fill(Output, mask, 0).to(conf.device)
            masked_outputs.append(Output.view(1,self.pad_len,88))
        masked_outputs = torch.cat(masked_outputs, 0)
        return masked_outputs

    def masking_labels(self, labels, n_padded):
        appended_labels = labels.clone().detach().view(1,-1).repeat(self.pad_len,1).transpose(0,1).to(conf.device)#.type(torch.float)
        masked_labels = []
        masks = torch.zeros(appended_labels.shape)
        for idx in range(len(n_padded)):
            masks[idx,self.pad_len-n_padded[idx]:] = 1
        masks = masks.type(torch.bool).to(conf.device)
        for label, pad, mask in zip(appended_labels, n_padded, masks):
            label = torch.masked_fill(label, mask, -100).to(conf.device)
            masked_labels.append(label)
        masked_labels = torch.stack(masked_labels)
        return masked_labels

    def compute_loss(self, inputs, outputs, labels, n_padded, w_rgr, w_clsf):
        rgrs, z, logits = outputs
        inputs = inputs.view(-1,self.pad_len,88)
        rgrs = rgrs.view(-1,self.pad_len,88)
        z = z.view(-1,self.pad_len,54)
        logits = logits.view(-1,self.pad_len,2)
        labels = labels-1#torch.nn.functional.one_hot(labels-1, num_classes=2)
        if self.target_model == 'rgrs':
            masked_outputs = self.masking_padded(rgrs, n_padded)
            masked_labels = self.masking_labels(labels, n_padded)
            try:
                loss = 0.85*self.rgrs_criterion(inputs, masked_outputs) + 0.15*self.clsf_criterion(logits.transpose(1,2), masked_labels.to(conf.device))
            except:
                pdb.set_trace()
        elif self.target_model == 'clsf':
            masked_labels = self.masking_labels(labels, n_padded)
            loss = self.clsf_criterion(logits.transpose(1,2), masked_labels.to(conf.device))
        elif self.target_model == 'joint':
            masked_outputs = self.masking_padded(rgrs, n_padded)
            masked_labels = self.masking_labels(labels, n_padded)
            loss = w_rgr*self.rgrs_criterion(inputs, masked_outputs) + w_clsf*self.clsf_criterion(logits.transpose(1,2), masked_labels.to(conf.device))
        return loss

    def save_model(self, epoch, model, optim):
        exp_name = f'ae-blstm-{self.target_model}-{now.month:03d}{now.day:03d}{now.hour:03d}{now.minute:03d}'
        try:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        }, f'{conf.exp_dir}/{exp_name}_ep{epoch:04d}.pt')
            print(f'model saved on {conf.exp_dir}/{exp_name}_ep{epoch:04d}.pt')
        except:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        }, f'{conf.exp_dir}/{exp_name}_ep{epoch:04d}.pt')
            print(f'model saved on {conf.exp_dir}/{exp_name}_ep{epoch:04d}.pt')

    def train(self):
        train_dataloader = self.train_dataloader
        valid_dataloader = self.valid_dataloader
        model = self.model
        patience = 0
        w_rgr, w_clsf = 1, 0
        for epoch in range(self.n_epoches):
            w_rgr = 1 * np.exp(-0.05 * epoch)
            w_clsf = 1 - w_rgr
            for train_batch in tqdm(train_dataloader, desc='training steps'):
                model.train()
                
                inputs, labels, n_padded = train_batch

                inputs = inputs.view(-1,88).to(conf.device)

                outputs = model(inputs)
                
                self.optimizer.zero_grad()
                try:
                    loss = self.compute_loss(inputs, outputs, labels, n_padded, w_rgr, w_clsf)
                except Exception as E:
                    print(E)
                    pdb.set_trace()
                    traceback.print_exc()
                loss.backward()
                self.optimizer.step()
            val_loss = self.validate(model, valid_dataloader, w_rgr, w_clsf)
            self.lr_scheduler.step(val_loss)
            if epoch == 0:
                min_val_loss = val_loss
            elif val_loss < min_val_loss:
                print(f'validation loss on epoch {epoch+1}, val_loss improved from {min_val_loss} to {val_loss}')
                min_val_loss = val_loss
                patience = 0
                self.save_model(epoch, model, self.optimizer)
            else:
                patience += 1
                if patience == 110:
                    break
                elif epoch == self.n_epoches - 1:
                    self.save_model('final', model, self.optimizer)
                    break

    def validate(self, model, valid_loader, w_rgr, w_clsf):
        model.eval()
        val_loss = []
        for val_batch in valid_loader:
            inputs, labels, n_padded = val_batch
            inputs = inputs.view(-1,88).to(conf.device)
            outputs = model(inputs)

            val_loss.append(self.compute_loss(inputs, outputs, labels, n_padded, w_rgr, w_clsf))

        return torch.mean(torch.tensor(val_loss))
