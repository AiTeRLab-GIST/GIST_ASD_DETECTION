import argparse
import numpy as np
import os
import sys
from transformers import AutoConfig, Wav2Vec2Processor
from scipy.io.wavfile import read as wavread
import torch
import conf
from model import Wav2Vec2ForSpeechClassification as Model
from utils import predict_probs

def predict_probs(wave, processor, model):
    features = processor(wave, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=False)

    input_values = features.input_values.to(conf.device)

    with torch.no_grad():
        logits = model(input_values).logits 

    return torch.nn.Softmax(1)(logits)

def inference(waveform):
    #load model
    exp_name = './exp/checkpoint'
    config = AutoConfig.from_pretrained(exp_name)
    model = Model.from_pretrained(exp_name, config=config).to(conf.device)
    processor = Wav2Vec2Processor.from_pretrained(conf.model_name)

    logits = predict_probs(waveform, processor, model)
    
    processor = Wav2Vec2Processor.from_pretrained('test')
    model = Model.from_pretrained(exp_name).to(conf.device)

    logits = predict_probs(batch, processor, model)

if __name__ == '__main__':
    sample_wav = np.random.randint(low = -32768, high = 32767, size = (16000,)).astype('float32')
    inference(sample_wav)