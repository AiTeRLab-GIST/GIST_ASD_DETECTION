# 파일명: load_datasets.py

import os
import conf  # conf.py 파일 임포트
from datasets import load_dataset, Dataset
from typing import Tuple, List

def load_datasets() -> Tuple[Dataset, Dataset, Dataset, List]:
    """
    conf.py에 정의된 경로를 기반으로 train, valid, test CSV 파일을 로드합니다.
    
    - CSV 파일은 탭(\t)으로 구분되어 있다고 가정합니다.
    - 데이터셋 정보와 레이블 통계를 출력합니다.

    Returns:
        Tuple[Dataset, Dataset, Dataset, List]:
            - train_dataset
            - valid_dataset
            - test_dataset
            - label_list (정렬된 고유 레이블 리스트)
    """
    
    # 1. conf.py를 참조하여 데이터 파일 경로를 안전하게 생성
    data_files = {
        "train": os.path.join(conf.SAVE_PATH, conf.DF_NAMES[0]),
        "valid": os.path.join(conf.SAVE_PATH, conf.DF_NAMES[1]),
        "test": os.path.join(conf.SAVE_PATH, conf.DF_NAMES[2]),
    }

    # 2. datasets 라이브러리를 사용하여 CSV 로드
    try:
        dataset_dict = load_dataset(
            "csv", 
            data_files=data_files, 
            delimiter="\t"
        )
    except FileNotFoundError as e:
        print(f"오류: 데이터셋 파일을 찾을 수 없습니다.")
        print(f"경로: {e.filename}")
        print("먼저 `dataset.py`를 실행하여 CSV 파일을 생성했는지 확인하세요.")
        raise SystemExit(1) # 프로그램 종료

    train_dataset = dataset_dict["train"]
    valid_dataset = dataset_dict["valid"]
    test_dataset = dataset_dict["test"]

    # 3. 로드된 데이터셋 정보 출력
    print("--- 데이터셋 로드 완료 ---")
    print(train_dataset)
    print(valid_dataset)
    print(test_dataset)
    print("-------------------------")

    # 4. 레이블 리스트 처리
    # 'asd' 대신 conf.py에 정의된 TARGET_COL_NAME 사용
    label_column = conf.TARGET_COL_NAME 
    
    try:
        label_list = train_dataset.unique(label_column)
        label_list.sort() # 레이블 정렬
        num_labels = len(label_list)
        print(f"타겟 컬럼: '{label_column}'")
        print(f"총 {num_labels}개 클래스: {label_list}")
    except Exception as e:
        print(f"오류: '{label_column}' 컬럼 처리 중 문제가 발생했습니다: {e}")
        print("conf.TARGET_COL_NAME이 CSV 헤더와 일치하는지 확인하세요.")
        raise SystemExit(1)

    return train_dataset, valid_dataset, test_dataset, label_list

# --- (선택 사항) 이 파일을 직접 실행하여 로드를 테스트할 수 있습니다 ---
if __name__ == '__main__':
    print("load_datasets() 함수 테스트 시작...")
    try:
        train_ds, valid_ds, test_ds, labels = load_datasets()
        print("\n[테스트 성공]")
        print(f"Train 샘플 수: {len(train_ds)}")
        print(f"Valid 샘플 수: {len(valid_ds)}")
        print(f"Test 샘플 수: {len(test_ds)}")
        print(f"레이블: {labels}")
    except Exception as e:
        print(f"\n[테스트 실패]: {e}")
        print("테스트를 위해서는 conf.py 설정이 올바르고,")
        print(f"'{conf.SAVE_PATH}' 디렉토리에 {conf.DF_NAMES} 파일들이 존재해야 합니다.")
