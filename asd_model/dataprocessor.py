import os
import sys
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset, load_metric
#from sklearn.model_selection import train_test_split

import conf
import pdb

from transformers import AutoConfig

input_column = conf.inputs
output_column = conf.outputs

class DataProcessor:
    def __init__(self, force_process = False):
        if not self.is_csv():
            print('no csv found')
            exit(-1)

    def is_csv(self):
        if os.path.isfile(f"{conf.save_path}/{conf.df_names[0]}") and os.path.isfile(f"{conf.save_path}/{conf.df_names[1]}"):
            return True
        else:
            return False

    def get_dataset_dataframe(self):
        data = []
        for path in tqdm(Path(conf.wav_path).glob("**/*.wav")):
            '''
            name = str(path).split('/')[-1].split('.')[0]
            label = str(path).split('/')[-2]
            '''
            label = str(path).split('_')[-2]
            if label == '3':
                continue
            else:
                name = str(path).split('/')[-1]
            
            try:
                # passing broken files
                s = torchaudio.load(path)
                data.append({
                    "name": name,
                    input_column: path,
                    output_column: label
                })
            except Exception as e:
                pass

        df = pd.DataFrame(data)
        df.head()
        print(f"Step 0: {len(df)}")
        df = self.filter_broken_data(df)
        return df

    def filter_broken_data(self, df):
        df["status"] = df[input_column].apply(lambda path: True if os.path.exists(path) else None)
        df = df.dropna(subset=[input_column])
        df = df.drop("status", 1)
        print(f"Step 1: {len(df)}")

        df = df.sample(frac=1)
        df = df.reset_index(drop=True)
        df.head()
        return df

    def load_dataset_and_config(self):
        data_files = {
            "train": f"{conf.save_path}/{conf.df_names[0]}",
            "validation": f"{conf.save_path}/{conf.df_names[1]}"
        }

        dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        print(train_dataset)
        print(eval_dataset)

        label_list = train_dataset.unique(output_column)
        label_list.sort()
        num_labels = len(label_list)
        print(f"A classification problem with {num_labels} classes: {label_list}")
        
        config = AutoConfig.from_pretrained(
        conf.model_name,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf")
        setattr(config, 'pooling_mode', conf.pooling_mode)
        setattr(config, 'mask_time_prob', 0)
        return train_dataset, eval_dataset, config, label_list
