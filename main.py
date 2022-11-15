import os
import numpy as np
import pdb
import torch
from tqdm import tqdm
#os.environ["NCCL_DEBUG"] = "INFO"
from functools import partial

import conf
from trainer import Trainer
#from dataprocessor import DataProcessor
import utils 
from dataset import egemaps_dataset as Dataset
from datasets import load_dataset, load_metric

from load_datasets import load_datasets

from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--featext', action = 'store_true')
parser.add_argument('--target_model', type=str, default = 'joint')

args = parser.parse_args()

device = torch.device('cuda')

def process():
    train_df, valid_df, test_df, label_list = load_datasets()
    
    if args.target_model == 'rgrs':
        from model import MultiTaskAutoEncoder as Model
        model = Model().cuda()
    if args.target_model == 'clsf':
        from model import AEBLSTMFT as Model
        model = Model().cuda()
        if args.train or args.featext:
            ae_exp = utils.get_part_model(part = 'rgrs')
            aedata = torch.load(ae_exp)
            model.AEPart.load_state_dict(aedata['model_state_dict'])
        elif args.eval:
            exp = utils.get_part_model(part = 'clsf')
            model_data = torch.load(exp)
            model.load_state_dict(model_data['model_state_dict'])
            print(f'Start evaluation on experiment:{exp}')
    if args.target_model == 'joint':
        from model import AEBLSTMJT as Model
        model = Model().cuda()
        if args.eval:
            exp = utils.get_part_model(part = 'joint')
            model_data = torch.load(exp)
            model.load_state_dict(model_data['model_state_dict'])
            print(f'Start evaluation on experiment:{exp}')

    train_dataset = Dataset(train_df, pad_len = 94)
    valid_dataset = Dataset(valid_df, pad_len = 94)
    test_dataset = Dataset(test_df, pad_len = 94)

    if args.train:
        if not os.path.isdir(conf.exp_dir):
            os.makedirs(conf.exp_dir, exist_ok=True)


        train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle=1)
        valid_dataloader = DataLoader(valid_dataset, batch_size = 16, shuffle=1)

        trainer = Trainer(model = model,
                          train_dataloader = train_dataloader,
                          valid_dataloader = valid_dataloader,
                          target_model = args.target_model,
                          pad_len = 94)
        trainer.train()

    if args.eval:
        model.eval()
        test_dataloader = DataLoader(test_dataset, batch_size = 16)
        import librosa
        from sklearn.metrics import classification_report
        test_dataset = load_dataset("csv", data_files={"test": f"{conf.save_path}/{conf.df_names[1]}"}, delimiter="\t")["test"]
        test_result = []
        true_labels = []
        results = []
        preds = []
        for test_batch in tqdm(test_dataloader, desc='evaluation steps'):
            inputs, labels, n_padded = test_batch
            inputs = inputs.view(-1,88)
            outputs = model(inputs.to(conf.device))
            for idx, output in enumerate(outputs[2].view(len(labels),-1,2)):
                output = output[:79-n_padded[idx],:]
                for outs in output:
                    if outs[0] > outs[1]:
                        results.append(0)
                    else:
                        results.append(1)
                if sum(results) >= len(labels)*0.5:
                    test_result.append(1)
                else:
                    test_result.append(0)
                results = []
            true_labels.append(labels)
        #test_result = np.concatenate(test_result, axis=0)
        pdb.set_trace()
        true_labels = np.concatenate(true_labels) - 1
        
        pdb.set_trace()
        report = classification_report(true_labels, test_result, digits=4)
        print(report)
    
    if args.featext:
        afeats_dict = {}
        alabels_dict = {}
        for idx in range(7):
            afeats_dict[idx] = []
            alabels_dict[idx] = []
        train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=0)
        exp = torch.load(utils.latest_exp())
        model = Model()
        model.load_state_dict(exp['model_state_dict'])
        for train_batch in tqdm(train_dataloader, desc='feat_ext steps'):
            inputs, labels, n_padded = train_batch        
            feats = model.forward(inputs, feat_ext = True)
            for idx, afeats in enumerate(feats):
                for feat, label in zip(afeats.detach().cpu().numpy(), labels):
                    afeats_dict[idx].append(feat)
                    alabels_dict[idx].append(label)
        for idx in range(7):
            np.save(os.path.join(conf.feat_ext_dir,f'feat_semi_{idx}.npy'), afeats_dict[idx])
            np.save(os.path.join(conf.feat_ext_dir,f'label_semi_{idx}.npy'), alabels_dict[idx])

if __name__ == '__main__':
    process()
