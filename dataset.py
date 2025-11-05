import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple

import torch
import torchaudio
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
## 1. eGeMAPS 피처 데이터셋 (e.g., pre-extracted features)
# -----------------------------------------------------------------------------

class egemaps_dataset(Dataset):
    """
    미리 추출된 eGeMAPS 피처(.csv)를 로드하고 패딩하는 데이터셋.
    DataFrame으로 'path' (피처 경로)와 'asd' (레이블) 컬럼을 받습니다.
    """
    def __init__(self, df: pd.DataFrame, pad_len: int = 94):
        self.df = df.reset_index(drop=True) # 인덱스 초기화
        self.pad_len = pad_len
        
        # 피처 파일의 피처 개수 (예: 88)
        # 임의의 파일 하나를 읽어 피처 수를 결정 (더 안전한 방법)
        try:
            sample_feat = pd.read_csv(self.df.at[0, 'path'], delimiter=';').values
            self.num_features = sample_feat.shape[1]
        except Exception:
            print("경고: 샘플 피처 로드 실패. 피처 수를 88로 가정합니다.")
            self.num_features = 88 # 또는 conf.NUM_FEATURES

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        반환: (패딩된 피처 텐서, 레이블 텐서, 패딩된 길이)
        """
        row = self.df.iloc[index]
        csv_path = row['path']
        label = row['asd']
        
        n_padded = 0 # 버그 수정: n_padded 초기화

        try:
            # .csv 또는 .txt 형태의 피처 로드
            try:
                feat = pd.read_csv(csv_path, delimiter=';').values
            except pd.errors.ParserError:
                # 버그 수정: 'path' -> 'csv_path'
                feat = np.loadtxt(csv_path, delimiter=';')
            
            feat = feat.astype(np.float32)

        except Exception as e:
            print(f"오류: {csv_path} 파일 로드 실패: {e}. 더미 텐서 반환.")
            feat = np.zeros((self.pad_len, self.num_features), dtype=np.float32)
            n_padded = self.pad_len # 전체가 패딩됨
            return (
                torch.from_numpy(feat).float(), 
                torch.tensor(label, dtype=torch.long), 
                n_padded
            )

        seq_len = feat.shape[0]

        if self.pad_len is not None:
            if seq_len < self.pad_len:
                # --- 패딩 (Shorter) ---
                n_padded = self.pad_len - seq_len
                feat = self._padding(feat, self.pad_len)
            
            elif seq_len > self.pad_len:
                # --- 절삭 (Longer) ---
                feat = feat[:self.pad_len, :]
                # n_padded는 0 유지

        return (
            torch.from_numpy(feat).float(), 
            torch.tensor(label, dtype=torch.long), 
            n_padded
        )

    def _padding(self, feat: np.ndarray, max_len: int) -> np.ndarray:
        """피처 배열의 끝(시간 축)에 제로 패딩을 추가합니다."""
        # (max_len, num_features) 크기의 0 배열 생성
        pad_len_feat = np.zeros((max_len, feat.shape[1]), dtype=np.float32)
        # 앞부분에 원본 피처 복사
        pad_len_feat[:feat.shape[0], :] = feat
        return pad_len_feat


# -----------------------------------------------------------------------------
## 2. Speech-to-Intent (S2I) 데이터셋 (e.g., raw audio)
# -----------------------------------------------------------------------------

def create_dict_from_text(file_path: str) -> Dict[str, str]:
    """
    (누락된 함수) 텍스트 파일에서 ID-텍스트 딕셔너리를 생성합니다.
    S2ITEXTDataset2가 이 함수를 필요로 합니다.
    
    예시 구현 (파일 형식에 맞게 수정 필요):
    """
    print(f"경고: {file_path}에서 create_dict_from_text를 로드합니다. (임시 구현)")
    cls_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1) # "ID text"
                if len(parts) == 2:
                    cls_dict[parts[0]] = parts[1]
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다. 빈 딕셔너리가 반환됩니다.")
        # raise NotImplementedError("create_dict_from_text 함수를 구현해야 합니다.")
    return cls_dict


class S2ITEXTDataset(Dataset):
    """
    오디오 경로(wav.scp)와 텍스트/레이블(text)을 병합하여 로드합니다.
    (wav_tensor, text), intent_class 를 반환합니다.
    """
    # 가독성을 위해 클래스 상수로 분리
    LABEL_TO_INT: Dict[str, int] = {'ASD': 1, 'TD': 0, 'OD': 2}

    def __init__(self, csv_path: str, wav_dir_path: str):
        # 안정성: 두 파일을 ID 기준으로 병합(merge)합니다. (단순 concat X)
        
        # 1. 텍스트 파일 로드 (예: "ID_001 some text ASD")
        try:
            df1 = pd.read_csv(csv_path, sep='\t', encoding='utf-8', header=None, names=['text_data'])
            # ID 추출 (첫 번째 공백 앞부분)
            df1['id'] = df1['text_data'].apply(lambda x: x.split(' ')[0])
        except Exception as e:
            print(f"오류: {csv_path} 로드 실패: {e}")
            raise
            
        # 2. 오디오 경로 파일 로드 (예: "ID_001 /path/to/wav.wav")
        try:
            wav_df = pd.read_csv(wav_dir_path, sep=' ', encoding='utf-8', header=None, names=['id', 'wav_path'])
        except Exception as e:
            print(f"오류: {wav_dir_path} 로드 실패: {e}")
            raise

        # 3. ID를 기준으로 두 DataFrame 병합
        self.df = pd.merge(df1, wav_df, on='id')
        print(f"S2ITEXTDataset: 총 {len(self.df)}개의 샘플 로드 완료.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, str], int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        
        # 텍스트 데이터 파싱 (예: "ID text text ... ASD")
        text_parts = row['text_data'].split(' ')
        intent_class_str = text_parts[-1]
        text = ' '.join(text_parts[1:-1])
        wav_path = row['wav_path']

        # 오디오 로드
        try:
            wav_tensor, _ = torchaudio.load(wav_path)
        except Exception as e:
            print(f"오류: {wav_path} 오디오 로드 실패: {e}. 더미 텐서 반환.")
            wav_tensor = torch.zeros(1, 16000) # 1초 길이의 더미

        # 레이블 변환
        intent_class = int(self.LABEL_TO_INT.get(intent_class_str, -1))
        if intent_class == -1:
            print(f"경고: 알 수 없는 레이블 '{intent_class_str}' (샘플 ID: {row['id']})")

        return (wav_tensor, text), intent_class


class S2ITEXTDataset2(S2ITEXTDataset):
    """
    S2ITEXTDataset을 상속받아, 추가적인 'cls_text'를 로드하고 반환합니다.
    (wav_tensor, cls_text, text), intent_class 를 반환합니다.
    """
    def __init__(self, csv_path: str, wav_dir_path: str):
        # 부모 클래스(__init__)를 호출하여 self.df를 먼저 생성
        super().__init__(csv_path, wav_dir_path)
        
        # 원본 코드의 경로 치환 로직 (매우 구체적이므로 그대로 유지)
        cls_dict_path = csv_path.replace('train_da_sum_sp', 'train_asr2cls_sp').replace('test_da', 'cls_test')
        
        # (누락되었던 함수) ID를 키로 하는 딕셔너리 로드
        self.cls_dict = create_dict_from_text(cls_dict_path)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, str, str], int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.df.iloc[idx]
        
        # 텍스트 데이터 파싱
        text_parts = row['text_data'].split(' ')
        intent_class_str = text_parts[-1]
        text = ' '.join(text_parts[1:-1])
        
        # 원본 코드의 경로 치환 로직 (wav_path)
        wav_path = row['wav_path'].replace('train_da_sum_sp', 'train_asr2cls_sp').replace('test_da', 'cls_test')
        
        # cls_text 가져오기
        sample_id = row['id']
        cls_text = self.cls_dict.get(sample_id, "") # ID가 없으면 빈 문자열
        
        if not cls_text:
            print(f"경고: cls_dict에서 ID '{sample_id}'를 찾을 수 없습니다.")

        # 원본 텍스트에서 cls_text 부분 제거
        text = text.replace(cls_text, '').strip()

        # 오디오 로드
        try:
            wav_tensor, _ = torchaudio.load(wav_path)
        except Exception as e:
            print(f"오류: {wav_path} 오디오 로드 실패: {e}. 더미 텐서 반환.")
            wav_tensor = torch.zeros(1, 16000)

        # 레이블 변환
        intent_class = int(self.LABEL_TO_INT.get(intent_class_str, -1))

        return (wav_tensor, cls_text, text), intent_class


# -----------------------------------------------------------------------------
## 3. 유틸리티 및 Collate 함수
# -----------------------------------------------------------------------------

def pad_or_truncate_list(
    input_list: List, 
    desired_length: int = 1500, 
    pad_value: Union[int, float] = 0
) -> List:
    """리스트를 원하는 길이로 패딩하거나 절삭합니다."""
    current_length = len(input_list)
    
    if current_length < desired_length:
        # 리스트가 짧은 경우: 패딩 추가
        return input_list + [pad_value] * (desired_length - current_length)
    else:
        # 리스트가 긴 경우: 절삭
        return input_list[:desired_length]


def collate_fn2(batch: List[Tuple[Tuple[torch.Tensor, str], int]]) \
    -> Tuple[Tuple[torch.Tensor, List[str]], torch.Tensor]:
    """
    S2ITEXTDataset용 collate 함수.
    (wav, text), label -> (padded_wavs, texts), labels
    """
    (seq, label) = zip(*batch)
    (wav, text) = zip(*seq)
    
    # [1, N] 또는 [N] 텐서를 [N]으로 평탄화
    seql = [wav_tensor.reshape(-1) for wav_tensor in wav]
    
    # 가변 길이의 오디오 시퀀스를 패딩
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    
    label = torch.tensor(list(label), dtype=torch.long)
    
    return (data, list(text)), label


def collate_fn4(batch: List[Tuple[Tuple[torch.Tensor, str, str], int]]) \
    -> Tuple[Tuple[torch.Tensor, List[str], List[str]], torch.Tensor]:
    """
    S2ITEXTDataset2용 collate 함수.
    (wav, cls_text, text), label -> (padded_wavs, cls_texts, texts), labels
    """
    (seq, label) = zip(*batch)
    (wav, cls_text, text) = zip(*seq)
    
    # [1, N] 또는 [N] 텐서를 [N]으로 평탄화
    seql = [wav_tensor.reshape(-1) for wav_tensor in wav]
    
    # 가변 길이의 오디오 시퀀스를 패딩
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    
    label = torch.tensor(list(label), dtype=torch.long)
    
    return (data, list(cls_text), list(text)), label
