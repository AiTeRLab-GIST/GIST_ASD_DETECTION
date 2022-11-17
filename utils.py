#utils
import os
import conf
import numpy as np
import glob

def get_part_model(part, fdir = './exp'):
    exps = []
    exp_paths = glob.glob(f'{fdir}/*/*.pt')
    for exp in sorted(exp_paths):
        if part in exp:
            exps.append(exp)
    return exps[-1]

def feat_ext(data, model):
    feats = model.forward(data, feat_ext = True)
    return feats
