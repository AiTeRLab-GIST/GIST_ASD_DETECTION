# 파일명: conf.py

import os
import torch
from datetime import datetime as dt

# -----------------------------------------------------------------------------
## 1. 경로 설정 (Path Configuration)
# -----------------------------------------------------------------------------

# --- 기본 데이터 경로 ---
BASE_DATA_DIR = './data/'
SAVE_PATH = BASE_DATA_DIR
DB_PATH = BASE_DATA_DIR
WAV_PATH = os.path.join(BASE_DATA_DIR, 'egemaps/')

# --- 데이터 파일 ---
# (load_datasets.py에서 사용될 것으로 예상)
DF_NAMES = ['train.csv', 'valid.csv', 'test.csv']

# --- 피처 추출 경로 ---
# (main.py --featext 실행 시 사용)
FEAT_EXT_DIR = '../feat_ext/ae'

# --- 실험 결과 (모델 가중치, 로그) 저장 경로 ---
# (Trainer.py에서 사용)
EXP_BASE_DIR = './exp/'


# -----------------------------------------------------------------------------
## 2. 데이터 설정 (Data Configuration)
# -----------------------------------------------------------------------------

SAMPLING_RATE = 16000

# eGeMAPS 데이터셋 관련 (dataset.py)
# (이전 코드의 'pad_len = 94'를 가져옴)
PAD_LEN = 94 

# (이전 코드의 'inputs = 'path', outputs = 'asd''를 명확하게 변경)
INPUT_COL_NAME = 'path'
TARGET_COL_NAME = 'asd'


# -----------------------------------------------------------------------------
## 3. 모델 및 실행 설정 (Model & Run Configuration)
# -----------------------------------------------------------------------------

# (이전 코드의 'condition = 'JT''를 가져옴)
# 실행할 모델의 조건이나 타입 (예: 'JT' (joint), 'rgrs', 'clsf')
CONDITION = 'JT' 

# (이전 코드의 'is_regression = False'를 가져옴)
IS_REGRESSION = False


# -----------------------------------------------------------------------------
## 4. 학습 하이퍼파라미터 (Training Hyperparameters)
# -----------------------------------------------------------------------------

# --- 장치 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 기본 학습 설정 ---
# (이전 코드의 'batch_size = 128'를 반영)
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128  # (평가 시에는 메모리가 허용하면 더 크게 설정 가능)

NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5 # (Optimizer L2 정규화)

# --- 로깅 및 저장 ---
# (Trainer.py에서 사용할 수 있는 설정값)
LOGGING_STEPS = 50    # 50 스텝마다 로그 출력
SAVE_TOTAL_LIMIT = 5  # 최대 5개의 체크포인트만 저장


# -----------------------------------------------------------------------------
## 5. 유틸리티 함수 (Utility Functions)
# -----------------------------------------------------------------------------

def get_experiment_dir():
    """
    현재 시간을 기준으로 고유한 실험 디렉토리 경로를 생성합니다.
    (예: ./exp/251105_1530)
    
    사용 예시 (main.py에서):
    exp_dir = conf.get_experiment_dir()
    os.makedirs(exp_dir, exist_ok=True)
    """
    now = dt.now()
    # YYYYMMDD_HHMM 형식
    dir_name = now.strftime('%y%m%d_%H%M') 
    return os.path.join(EXP_BASE_DIR, dir_name)
