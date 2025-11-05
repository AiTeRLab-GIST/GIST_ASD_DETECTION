# 파일명: dataset.py

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split

import conf  # conf.py의 설정값들을 가져옴

# conf.py의 대문자 상수를 사용
INPUT_COLUMN = conf.INPUT_COL_NAME
OUTPUT_COLUMN = conf.TARGET_COL_NAME


class DataProcessor:
    """
    원본 eGeMAPS .csv 피처 파일들을 스캔하여
    학습/검증/테스트용 메타데이터 CSV 파일 3개를 생성합니다.
    """
    def __init__(self):
        print("DataProcessor가 초기화되었습니다.")
        # __init__ 에서는 무거운 로직을 실행하지 않습니다.

    def is_csv_present(self):
        """train, valid, test CSV 파일이 모두 존재하는지 확인합니다."""
        train_file = os.path.join(conf.SAVE_PATH, conf.DF_NAMES[0])
        valid_file = os.path.join(conf.SAVE_PATH, conf.DF_NAMES[1])
        test_file = os.path.join(conf.SAVE_PATH, conf.DF_NAMES[2])
        
        if os.path.isfile(train_file) and os.path.isfile(valid_file) and os.path.isfile(test_file):
            return True
        else:
            return False

    def get_dataset_dataframe(self):
        """
        conf.WAV_PATH (eGeMAPS 피처 경로)를 재귀적으로 탐색하여
        파일 경로와 레이블을 DataFrame으로 만듭니다.
        """
        data = []
        # .csv로 된 피처 파일을 찾음
        file_paths = list(Path(conf.WAV_PATH).glob("**/*.csv"))
        print(f"총 {len(file_paths)}개의 .csv 피처 파일을 탐색합니다...")

        for path in tqdm(file_paths, desc="피처 파일 스캔 중"):
            try:
                # 파일명 규칙에 의존하여 레이블 추출 (예: ..._asd_1_...)
                # 이 부분은 사용자의 파일명 규칙에 따라 수정이 필요할 수 있습니다.
                label = str(path.stem).split('_')[-2]

                if label == '3': # '3'번 레이블은 제외
                    continue
                
                name = path.name

                # 비어있는 파일이나 잘못된 파일을 거름
                # header=None: 헤더가 없음, delimiter=';': 세미콜론 구분
                feat = pd.read_csv(path, header=None, delimiter=';')
                
                if len(feat.values) > 1: # 최소 2 프레임 이상인 데이터만
                    data.append({
                        "name": name,
                        INPUT_COLUMN: str(path), # 'path' 컬럼에 파일 전체 경로
                        OUTPUT_COLUMN: int(label) # 'asd' 컬럼에 레이블
                    })
                else:
                    print(f"경고: {name} 파일이 비어있거나 데이터가 1줄뿐이라 건너뜁니다.")
            
            except Exception as e:
                print(f"오류: {path} 처리 중 문제 발생 ({e}). 건너뜁니다.")

        df = pd.DataFrame(data)
        print(f"유효한 데이터 {len(df)}개 수집 완료.")
        return df

    def save_db_list_to_csv(self, df):
        """
        DataFrame을 train, valid, test 셋으로 분할하고 CSV 파일로 저장합니다.
        (예: 80% train, 10% valid, 10% test)
        """
        # 1단계: Train(80%) / Temp(20%) 분리
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=101, 
            stratify=df[OUTPUT_COLUMN] # 레이블 분포를 유지하며 분리
        )
        
        # 2단계: Temp(20%) -> Valid(10%) / Test(10%) 분리
        valid_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            random_state=101, 
            stratify=temp_df[OUTPUT_COLUMN]
        )

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # conf.py에 정의된 파일명으로 저장
        train_df.to_csv(os.path.join(conf.SAVE_PATH, conf.DF_NAMES[0]), sep="\t", encoding="utf-8", index=False)
        valid_df.to_csv(os.path.join(conf.SAVE_PATH, conf.DF_NAMES[1]), sep="\t", encoding="utf-8", index=False)
        test_df.to_csv(os.path.join(conf.SAVE_PATH, conf.DF_NAMES[2]), sep="\t", encoding="utf-8", index=False)

        print("\n--- 데이터 분할 및 저장 완료 ---")
        print(f"Train: {train_df.shape}")
        print(f"Valid: {valid_df.shape}")
        print(f"Test : {test_df.shape}")

    def run(self, force_process=False):
        """
        메인 실행 함수.
        CSV 파일이 없거나 force=True일 때만 데이터 처리를 실행합니다.
        """
        if not self.is_csv_present() or force_process:
            if force_process:
                print("--force 옵션이 감지되었습니다. 데이터를 강제로 재생성합니다.")
            else:
                print("CSV 파일이 존재하지 않습니다. 데이터 처리를 시작합니다.")
            
            df = self.get_dataset_dataframe()
            if len(df) > 0:
                self.save_db_list_to_csv(df)
            else:
                print("오류: 처리할 유효한 데이터가 없습니다. conf.WAV_PATH를 확인하세요.")
        else:
            print("CSV 파일이 이미 존재합니다. 처리를 건너뜁니다.")
            print(f"(재생성하려면 'python {__file__} --force'를 실행하세요)")


# -----------------------------------------------------------------------------
## PyTorch Dataset 클래스 (main.py에서 임포트하여 사용)
# -----------------------------------------------------------------------------

class egemaps_dataset(TorchDataset):
    """
    Pandas DataFrame을 입력받아 PyTorch DataLoader에서 사용할 수 있도록
    eGeMAPS 피처(.csv)를 로드하고 패딩하는 PyTorch Dataset 클래스입니다.
    """
    def __init__(self, dataframe, pad_len):
        self.df = dataframe
        self.pad_len = pad_len
        self.input_col = INPUT_COLUMN
        self.target_col = OUTPUT_COLUMN
        
        # eGeMAPS 피처 개수 (88개)
        # (main.py의 `inputs.view(-1, 88)`에서 88을 가져옴)
        self.num_features = 88 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        csv_path = row[self.input_col]
        label = row[self.target_col]

        try:
            # eGeMAPS 피처 로드 (header=None, delimiter=';')
            feats = pd.read_csv(csv_path, header=None, delimiter=';').values
            feats = feats.astype(np.float32)
        except Exception as e:
            print(f"오류: {csv_path} 파일 로드 실패: {e}")
            # 오류 발생 시 더미 데이터 반환
            feats = np.zeros((self.pad_len, self.num_features), dtype=np.float32)
            n_padded = self.pad_len
            return (
                torch.tensor(feats, dtype=torch.float32), 
                torch.tensor(label, dtype=torch.long), 
                n_padded
            )

        n_padded = 0
        seq_len = feats.shape[0]

        if seq_len < self.pad_len:
            # --- 패딩 (Shorter) ---
            pad_width = self.pad_len - seq_len
            n_padded = pad_width
            # (pad_width, 0) -> 아래쪽에 패딩, (0, 0) -> 피처 차원은 패딩 안 함
            feats = np.pad(feats, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
            
        elif seq_len > self.pad_len:
            # --- 절삭 (Longer) ---
            feats = feats[:self.pad_len, :]
        
        # main.py에서 (inputs, labels, n_padded) 튜플을 사용하므로 동일하게 반환
        return (
            torch.tensor(feats, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.long), 
            n_padded
        )

# -----------------------------------------------------------------------------
## 이 파일을 직접 실행할 경우 (데이터 전처리)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="eGeMAPS 피처를 스캔하여 train/valid/test.csv를 생성합니다."
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help="CSV 파일이 이미 존재하더라도 강제로 재생성합니다."
    )
    args = parser.parse_args()

    # 데이터 처리기 실행
    processor = DataProcessor()
    processor.run(force_process=args.force)
