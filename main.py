import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import argparse

# sklearn, librosa 등 모든 import는 파일 상단에 위치시키는 것이 좋습니다.
from sklearn.metrics import classification_report
import librosa
from datasets import load_dataset

# 프로젝트 로컬 모듈
import conf
from trainer import Trainer
import utils
from dataset import egemaps_dataset as Dataset # 'Dataset'으로 alias
from load_datasets import load_datasets
# 모델 임포트는 load_model_for_task 함수 내부로 이동시켜 필요한 모델만 로드하도록 함

# --- 1. 전역 설정 ---
# 하드코딩된 값을 상수로 관리
PAD_LEN = 94
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16 # 학습과 평가의 배치 사이즈는 다를 수 있습니다.

def setup_arg_parser():
    """커맨드 라인 인자를 파싱하는 함수를 분리합니다."""
    parser = argparse.ArgumentParser(description="eGeMAPS 모델 학습 및 평가")
    parser.add_argument('--train', action='store_true', help="모델을 학습 모드로 실행합니다.")
    parser.add_argument('--eval', action='store_true', help="모델을 평가 모드로 실행합니다.")
    parser.add_argument('--featext', action='store_true', help="피처를 추출합니다.")
    parser.add_argument('--target_model', type=str, default='joint', 
                        choices=['rgrs', 'clsf', 'joint'], 
                        help="사용할 모델 타입을 선택합니다 (rgrs, clsf, joint)")
    return parser.parse_args()

def load_model_for_task(args, device):
    """
    args에 따라 적절한 모델을 임포트하고,
    작업(train, eval, featext)에 맞게 가중치를 로드합니다.
    """
    model = None
    ModelClass = None

    if args.target_model == 'rgrs':
        from model import MultiTaskAutoEncoder as ModelClass
        model = ModelClass().to(device)
        # rgrs 모델은 원본 코드에서 별도의 가중치 로딩 로직이 없었습니다.
        # (필요시 추가)

    elif args.target_model == 'clsf':
        from model import AEBLSTMFT as ModelClass
        model = ModelClass().to(device)
        
        if args.train or args.featext:
            # 학습 또는 피처 추출 시 AE 부분 가중치 선행 로드
            print("clsf 모드: rgrs(AE) 가중치를 사전 로드합니다.")
            ae_exp_path = utils.get_part_model(part='rgrs')
            if ae_exp_path:
                aedata = torch.load(ae_exp_path, map_location=device)
                model.AEPart.load_state_dict(aedata['model_state_dict'])
            else:
                print("경고: 사전 로드할 rgrs 모델을 찾지 못했습니다.")
                
        elif args.eval:
            # 평가 시 clsf 전체 모델 가중치 로드
            exp_path = utils.get_part_model(part='clsf')
            if exp_path:
                model_data = torch.load(exp_path, map_location=device)
                model.load_state_dict(model_data['model_state_dict'])
                print(f'clsf 평가 시작. 로드된 모델: {exp_path}')
            else:
                print("오류: 평가할 clsf 모델을 찾지 못했습니다.", flush=True)
                return None

    elif args.target_model == 'joint':
        from model import AEBLSTMJT as ModelClass
        model = ModelClass().to(device)
        
        if args.eval:
            # 평가 시 joint 전체 모델 가중치 로드
            exp_path = utils.get_part_model(part='joint')
            if exp_path:
                model_data = torch.load(exp_path, map_location=device)
                model.load_state_dict(model_data['model_state_dict'])
                print(f'joint 평가 시작. 로드된 모델: {exp_path}')
            else:
                print("오류: 평가할 joint 모델을 찾지 못했습니다.", flush=True)
                return None
                
    if model is None:
        print(f"오류: {args.target_model}에 대한 모델을 로드하지 못했습니다.")
        
    return model, ModelClass # ModelClass는 featext에서 재사용될 수 있음

def run_training(model, train_dataset, valid_dataset, args):
    """모델 학습 로직"""
    print("--- 모델 학습 시작 ---")
    if not os.path.isdir(conf.exp_dir):
        os.makedirs(conf.exp_dir, exist_ok=True)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=TRAIN_BATCH_SIZE, 
                                  shuffle=True) # 1 대신 True 사용
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=EVAL_BATCH_SIZE, 
                                  shuffle=False) # 검증셋은 보통 shuffle하지 않음

    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      target_model=args.target_model,
                      pad_len=PAD_LEN)
    trainer.train()
    print("--- 모델 학습 완료 ---")

def run_evaluation(model, test_dataset, device):
    """모델 평가 로직"""
    print("--- 모델 평가 시작 ---")
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)

    # 원본 코드의 변수명 사용
    test_result = []
    true_labels = []

    for test_batch in tqdm(test_dataloader, desc="평가 단계"):
        inputs, labels, n_padded = test_batch
        
        # 원본 로직 유지 (입력 차원 및 모델 출력 처리)
        # [B, 94, 88] -> [B*94, 88] (?) - 원본 코드의 view(-1, 88)
        # 이 부분은 모델의 forward를 확인해야 함. [B, 94*88]일 수도 있음.
        # 원본 코드 `inputs = inputs.view(-1, 88)`를 존중.
        inputs = inputs.view(-1, 88).to(device)

        with torch.no_grad():
            outputs = model(inputs) # device는 model이 가지고 있음

        # 출력(outputs[2]) 처리 로직 (원본 코드 기준)
        # (B, 79, 2) 크기로 추정됨
        batch_preds = outputs[2].view(len(labels), -1, 2) 
        
        for idx, output_seq in enumerate(batch_preds):
            # 패딩 제외
            valid_len = 79 - n_padded[idx] # 79는 모델 출력 길이와 관련?
            valid_output = output_seq[:valid_len, :]
            
            # 프레임 레벨 예측
            frame_results = (valid_output[:, 1] > valid_output[:, 0]).int().cpu().numpy()
            
            # 다수결 투표 (Majority Voting)
            if np.sum(frame_results) >= (len(frame_results) * 0.5):
                test_result.append(1)
            else:
                test_result.append(0)
        
        true_labels.append(labels.cpu().numpy())

    true_labels = np.concatenate(true_labels) - 1 # 라벨 1-based -> 0-based
    
    print("\n--- 평가 결과 ---")
    report = classification_report(true_labels, test_result, digits=4)
    print(report)

def run_feature_extraction(train_dataset, args, ModelClass, device):
    """
    피처 추출 로직.
    이 함수는 원본 코드처럼 별개의 모델을 로드하여 피처를 추출합니다.
    (주의: main에서 로드한 모델과 다른 모델임)
    """
    print("--- 피처 추출 시작 ---")
    
    # 1. 원본 코드의 로직: `latest_exp`에서 모델을 새로 로드
    # (주의: args.target_model에 맞는 ModelClass가 필요)
    try:
        exp_path = utils.latest_exp() # 이 함수가 utils.py에 있다고 가정
        exp = torch.load(exp_path, map_location=device)
        
        model = ModelClass().to(device)
        model.load_state_dict(exp['model_state_dict'])
        model.eval()
        print(f"피처 추출 모델 로드: {exp_path}")
    except Exception as e:
        print(f"오류: 피처 추출 모델 로드 실패 ({e}).")
        print("`utils.latest_exp()` 또는 `ModelClass`가 `args.target_model`과 일치하는지 확인하세요.")
        return

    # 2. 데이터 준비
    afeats_dict = {idx: [] for idx in range(7)} # 7은 아마도 레이어 수?
    alabels_dict = {idx: [] for idx in range(7)}
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # 3. 피처 추출 루프
    for train_batch in tqdm(train_dataloader, desc="피처 추출 단계"):
        inputs, labels, n_padded = train_batch
        inputs = inputs.to(device) # [B, 94, 88]
        
        with torch.no_grad():
            # 원본 코드의 forward 호출 방식
            feats_per_layer = model.forward(inputs, feat_ext=True) 

        # feats_per_layer는 리스트 또는 튜플 (길이 7)
        for layer_idx, afeats in enumerate(feats_per_layer):
            # afeats: [B, Seq_len, Feat_dim]
            for feat_seq, label in zip(afeats.detach().cpu().numpy(), labels):
                afeats_dict[layer_idx].append(feat_seq)
                alabels_dict[layer_idx].append(label)

    # 4. 저장
    if not os.path.isdir(conf.feat_ext_dir):
        os.makedirs(conf.feat_ext_dir, exist_ok=True)
        
    for idx in range(7):
        feat_path = os.path.join(conf.feat_ext_dir, f'feat_semi_{idx}.npy')
        label_path = os.path.join(conf.feat_ext_dir, f'label_semi_{idx}.npy')
        
        np.save(feat_path, np.array(afeats_dict[idx]))
        np.save(label_path, np.array(alabels_dict[idx]))
    
    print(f"--- 피처 추출 완료. {conf.feat_ext_dir}에 저장됨 ---")

def main():
    """메인 실행 함수"""
    args = setup_arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    print(f"실행 작업: Train({args.train}), Eval({args.eval}), FeatExt({args.featext})")
    print(f"타겟 모델: {args.target_model}")

    # --- 1. 데이터셋 로드 ---
    # 이 부분은 모든 작업에 공통적으로 필요
    print("데이터셋 로드 중...")
    try:
        train_df, valid_df, test_df, label_list = load_datasets()
        
        train_dataset = Dataset(train_df, pad_len=PAD_LEN)
        valid_dataset = Dataset(valid_df, pad_len=PAD_LEN)
        test_dataset = Dataset(test_df, pad_len=PAD_LEN)
        print("데이터셋 로드 완료.")
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return

    # --- 2. 모델 로드 ---
    # (학습/평가용 모델. 피처 추출은 별도 모델을 로드함)
    model, ModelClass = load_model_for_task(args, device)
    if model is None and (args.train or args.eval):
        print("모델 초기화에 실패하여 프로그램을 종료합니다.")
        return
    if ModelClass is None:
        print("모델 클래스를 가져오지 못했습니다. (featext가 실패할 수 있음)")

    # --- 3. 작업 실행 ---
    if args.train:
        run_training(model, train_dataset, valid_dataset, args)

    if args.eval:
        run_evaluation(model, test_dataset, device)

    if args.featext:
        # 원본 코드의 로직을 따라, 피처 추출은 별개의 함수로 분리
        # (이 함수는 내부적으로 `utils.latest_exp()`를 통해 *다른* 모델을 로드함)
        if ModelClass is None:
            # rgrs 모드일 때 ModelClass가 None일 수 있으므로 재로드
            if args.target_model == 'rgrs':
                from model import MultiTaskAutoEncoder as ModelClass
            else:
                print("오류: 피처 추출을 위한 모델 클래스를 알 수 없습니다.")
                return
                
        run_feature_extraction(train_dataset, args, ModelClass, device)

if __name__ == '__main__':
    process() # 원본 코드의 함수명 유지
