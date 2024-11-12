from . import conf
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model)

from dataclasses import dataclass
from typing import Optional, Tuple

class dense_block(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.linear = nn.Linear(in_shape, out_shape)
        self.BN = nn.BatchNorm1d(out_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if len(x.shape) == 3:
            x = self.BN(x.permute(0,2,1)).permute(0,2,1)
        else:
            x = self.BN(x)
        x = self.relu(x)
        return x

class lstm_block(nn.Module):
    def __init__(self, in_shape, hidden_size, batch_first):
        super().__init__()
        self.lstm = nn.LSTM(in_shape, hidden_size, num_layers = 1, batch_first = True, bidirectional = True)
        self.BN = nn.BatchNorm1d(2 * hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.BN(x.permute(0,2,1)).permute(0,2,1)
        return x


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = dense_block(config.hidden_size, config.hidden_size)
        self.blstm = lstm_block(config.hidden_size, 128, True) 
        self.dense2 = dense_block(256, 128)
        self.dense3 = dense_block(128, 64)
        self.dense4 = dense_block(64, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dense1(x)
        x = self.blstm(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = x.permute(0,2,1)
        x = nn.MaxPool1d(x.shape[-1])(x).squeeze()
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_feature=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        #hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(torch.tensor([2.0, 1.0]))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss().to(conf.device)
                try:
                    loss = loss_fct(logits, labels)
                except:
                    labels_temp = np.zeros((logits.shape[0], logits.shape[1]))
                    for idx, label in enumerate(labels.cpu().numpy()):
                        labels_temp[idx, int(label)] = 1
                    labels = torch.from_numpy(labels_temp).to(conf.device)
                    loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        elif return_feature:
            return hidden_states

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
