import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split

import conf

from transformers import AutoConfig

input_column = conf.inputs
output_column = conf.outputs

class DataProcessor:
    def __init__(self, force_process = False):
        if not self.is_csv() or force_process:
            df = pd.read_csv(f'{conf.save_path}/conf.df_names[0]}')#self.get_dataset_dataframe()
            self.save_db_list_to_csv(df)

    def is_csv(self):
        if os.path.isfile(f"{conf.save_path}/{conf.df_names[0]}") and os.path.isfile(f"{conf.save_path}/{conf.df_names[2]}"):
            return True
        else:
            return False

    def get_dataset_dataframe(self):
        data = []
        for path in tqdm(Path(conf.wav_path).glob("**/*.csv")):
            label = str(path).split('_')[-2]
            if label == '3':
                continue
            else:
                name = str(path).split('/')[-1]
            # passing empty files
            try:
                feat = pd.read_csv(path, header=None, delimiter = ';')
                if len(feat.values) > 1:
                    data.append({
                        "name": name,
                        input_column: path,
                        output_column: label
                    })
                else:
                    pass
            except:
                try:
                    feat = np.loadtxt(path, delimiter =';')
                    if len(feat.values) > 1:
                        data.append({
                            "name": name,
                            input_column: path,
                            output_column: label
                        })
                    else:
                        pass
                except:
                    pass

        df = pd.DataFrame(data)
        df.head()

        print(f"Step 0: {len(df)}")
        return df

    def save_db_list_to_csv(self, df):
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=101, stratify=df[conf.outputs])
        #valid_df, test_df = test_df[:int(0.5*len(test_df))], test_df[int(0.5*len(test_df)):]

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        #test_df = test_df.reset_index(drop=True)

        train_df.to_csv(f"{conf.save_path}/{conf.df_names[0]}", sep="\t", encoding="utf-8", index=False)
        valid_df.to_csv(f"{conf.save_path}/{conf.df_names[1]}", sep="\t", encoding="utf-8", index=False)
        #test_df.to_csv(f"{conf.save_path}/{conf.df_names[2]}", sep="\t", encoding="utf-8", index=False)

        print(train_df.shape)
        print(valid_df.shape)
        #print(test_df.shape)

    def load_datasets(self, data_set=None):
        if type(data_set) != type(list()):
            if type(data_set) == type(str()):
                data_set.append(data_set)
            else:
                print('invalid data set notation')
                exit(-1)
                
        data_files = {
            "train": f"{conf.save_path}/{conf.df_names[0]}",
            "valid": f"{conf.save_path}/{conf.df_names[1]}",
            "test": f"{conf.save_path}/{conf.df_names[2]}"
        }

        dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
        train_dataset = dataset["train"]
        valid_dataset = dataset["valid"]
        test_dataset = dataset["test"]

        print(train_dataset)
        print(valid_dataset)
        print(test_dataset)

        label_list = train_dataset.unique(output_column)
        label_list.sort()
        num_labels = len(label_list)
        print(f"A classification problem with {num_labels} classes: {label_list}")
        
        return train_dataset, valid_dataset, test_dataset, label_list
