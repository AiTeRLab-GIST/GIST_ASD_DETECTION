import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
import json
import os
import numpy as np
import onnxruntime
import triton_python_backend_utils as pb_utils
from pyannote.audio import Pipeline
from espnet_onnx import Speech2Text
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, BertTokenizerFast, AutoTokenizer, AutoConfig
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor, as_completed
from asd_model.model import Wav2Vec2ForSpeechClassification as Model
from combined_model.model import MMCATextModel2

class TritonPythonModel:
    def initialize(self, args):
        # 모델 및 파이프라인 초기화
        path = os.path.join(args["model_repository"], args["model_version"])
        self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="YOUR_HUGGINGFACE_TOKEN")
        HYPER_PARAMETERS = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.5,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.5
        }
        self.pipeline.instantiate(HYPER_PARAMETERS)
        exp_name = path+'/asd_model/exp/checkpoint'
        self.config = AutoConfig.from_pretrained(exp_name)
        self.model = Model.from_pretrained(exp_name, config=self.config)
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.ged_tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.ged_session = onnxruntime.InferenceSession(path+"/ged_kor.onnx")
        self.dac_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.dac_session = onnxruntime.InferenceSession(path+"/dac_kor.onnx")
        self.asr = Speech2Text(model_dir=path+'/model/child')
        self.combined = MMCATextModel2(exp_name=exp_name, da=path+"/combined_model/model_swda_kor.pt")
        state_dict = torch.load(path+"/combined_model/combined_asd.pth")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "", 1) if key.startswith("model.") else key  # 모델에 따라 조정
            new_key = new_key.replace("module.", "")  # 'module.' 접두어가 있다면 제거
            new_state_dict[new_key] = value
        self.combined.load_state_dict(new_state_dict)
        self.sampling_rate = 16000
        self.frame_size = 2048
        self.label_map = {
            0: '선언', 1: '단순 진술', 2: '주장', 3: '수용/긍정', 4: '거절/부정', 
            5: '답변 회피', 6: '명령', 7: '제안/요청', 8: '단순 질문', 9: '확인 질문', 
            10: '약속', 11: '경고/협박', 12: '감사', 13: '사과', 14: '감탄', 
            15: '칭찬', 16: '비난/불평', 17: '슬픔/괴로움', 18: '기타 표출', 
            19: '첫인사', 20: '끝인사', 21: '기타 인사', 22: '부름'
        }


    def time_range_to_frame_size(self, start_seconds, end_seconds, sampling_rate):
        frame_size_start = int(start_seconds * sampling_rate)
        frame_size_end = int(end_seconds * sampling_rate)
        return (frame_size_start, frame_size_end)
    
    def asr_inference(self, input_data):
        nbest = self.asr(input_data)
        return nbest[0][0]
    
    def combined_inference(self, input_tuple):
        audio_tensor, asr_result = input_tuple
        if audio_tensor is None or asr_result is None:
            print("Error: none")
            return None
        
        nbest = self.combined(input_tuple)
        return F.sigmoid(nbest[0][0]).detach().cpu().numpy().tolist()
    
    def ged_inference(self, sentence):
        ged_tokens = self.ged_tokenizer(sentence, return_tensors="np")
        onnx_result = self.ged_session.run(None, {'input_ids': ged_tokens['input_ids'], 
                                                  'attention_mask': ged_tokens['attention_mask']})
        onnx_pred_class = np.argmax(onnx_result[0], axis=1)[0]
        onnx_pred_label = "상동행동 없음" if onnx_pred_class == 0 else "상동행동 있음"
        return onnx_pred_label
    
    def dac_inference(self, sentence):
        inputs = self.dac_tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = torch.tensor(inputs.input_ids.numpy()).unsqueeze(1)
        attention_mask = torch.tensor(inputs.attention_mask.numpy()).unsqueeze(1)
        chunk_lens = np.array([1], dtype=np.int64)
        onnx_outputs = self.dac_session.run(None, {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
            'chunk_lens': chunk_lens
        })
        onnx_pred_class = np.argmax(onnx_outputs[0], axis=1)[0]
        onnx_pred_label = self.label_map[onnx_pred_class]
        return onnx_pred_label
    
    def execute(self, requests):
        responses = []
        
        for request in requests:
            input_audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_INPUT").as_numpy()
            
            audio_data = input_audio_tensor.squeeze(0)
            input_audio_tensor = torch.Tensor(input_audio_tensor)
                        
            output = self.pipeline({"waveform": input_audio_tensor, "sample_rate": self.sampling_rate})
            segments = [self.time_range_to_frame_size(speech.start, speech.end, self.sampling_rate) for speech in output.get_timeline().support()]
            asr_output = []
            for start, end in segments:
                asr_output.append(self.asr_inference(audio_data[start:end]))
            with ThreadPoolExecutor(max_workers=10) as executor:
                                
                # Submitting GED and DAC inferences for each ASR result
                ged_futures = {executor.submit(self.ged_inference, asr_result): asr_result for asr_result in asr_output}
                dac_futures = {executor.submit(self.dac_inference, asr_result): asr_result for asr_result in asr_output}

                ged_output = [future.result() for future in as_completed(ged_futures)]
                dac_output = [future.result() for future in as_completed(dac_futures)]
                
                # Submitting Combined Inference for each segment
                combined_futures = [executor.submit(self.combined_inference, (input_audio_tensor[None, 0, start:end], asr_result)) for (start, end), asr_result in zip(segments, asr_output)]
                combined_output = [future.result() for future in as_completed(combined_futures)]
                
            response_dict = {'asd_output': combined_output, 'final_asd_output': (sum(combined_output)/len(combined_output)), 'segments': [[start / self.sampling_rate, end / self.sampling_rate] for start, end in segments], 'ged_output': ged_output, 'dac_output': dac_output}
            response_json = json.dumps(response_dict).encode('utf-8')
            
            response_tensor = pb_utils.Tensor("OUTPUT_DICT", np.array([response_json], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse([response_tensor]))
            
        return responses
