#utils
import torch
import pdb
import conf
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, EvalPrediction

def get_exp(exp_name):
    exps = os.listdir(exp_name)
    exp_idx = np.argmax([exp.split('-')[1] for exp in exps])
    return exps[exp_idx]

def speech_file_to_array_fn(path):
    #pdb.set_trace()
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, conf.sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def speech_batch_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    #speech_array = torchaudio.transforms.Resample(np.asarray(speech_array), sampling_rate, conf.sampling_rate)
    speech_array = np.asarray(speech_array)

    batch["speech"] = speech_array
    return batch

def predict(batch, processor, model):
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(conf.device)
    #attention_mask = features.attention_mask.to(conf.device)

    with torch.no_grad():
        #logits = model(input_values, attention_mask=attention_mask).logits 
        logits = model(input_values).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

def predict_probs(wave, processor, model):
    features = processor(wave, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=False)

    input_values = features.input_values.to(conf.device)

    with torch.no_grad():
        logits = model(input_values).logits 

    return torch.nn.Softmax(1)(logits)

def feat_ext(batch, processor, model):
    #pdb.set_trace()
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(conf.device)
    feat = model.forward(input_values = input_values, return_feature = True)
    label = batch["asd"]
    return feat, label

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples,label_list, processor):
    speech_list = [speech_file_to_array_fn(path) for path in examples[conf.inputs]]
    target_list = [label_to_id(label, label_list) for label in examples[conf.outputs]]

    result = processor(speech_list, sampling_rate=conf.sampling_rate)
    #pdb.set_trace()
    result["labels"] = list(target_list)

    return result

def compute_metrics(p: EvalPrediction):
    #pdb.set_trace()
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if conf.is_regression else np.argmax(preds, axis=1)

    if conf.is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
