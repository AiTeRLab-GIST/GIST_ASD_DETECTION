# GIST_ASD_DETECTION
Deep learning based autism spectral disorder detection from children voice

# 과제 개요


# 프로그램 설명
영유아 음성 데이터로부터 추출된 음성 특징 입력에 대해, 영유아의 ASD/TD 여부를 판정하기 위한 프로그램

# 코드 설명
conf.py: configuration 파일
dataprocessor.py: 데이터 처리를 위한 모듈
dataset.py: 데이터 입력 양식을 위한 모듈
load_datasets.py: 처리된 데이터셋을 로드하기 위한 모듈
main.py: 모델 훈련 및 평가를 위한 소스코드
model.py: 딥러닝 모델을 불러오기 위한 모듈
trainer.py: 딥러닝 모델 훈련을 위한 모듈
utils.py: 기타 함수 모듈

# arguments
main.py [--train] [--eval] [--target_model rgrs, clsf, joint]

# Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-00330, Development of AI Technology for Early Screening of Child/Child Autism Spectrum Disorders based on Cognition of the Psychological Behavior and Response).
