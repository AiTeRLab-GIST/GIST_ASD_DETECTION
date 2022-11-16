# GIST_ASD_DETECTION
영유아 음성 특징 기반 자폐스펙트럼 추정 알고리즘

# 기술 개요
영유아 및 아동의 옹알이성 발성 및 비전형적인 발성으로 부터 음성 특징을 추출하고 ASD/TD 영유아 그룹간의 음성 특징 차이 분석 및 인공 지능 기반의 알고리즘을 이용하여 ASD/TD 영유아 그룹 간 음성 특징을 분석하는 기술의 일환으로, Auto-encoder 기반 특징 추출 및 ASD/TD 영유아 판별

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

utils.py: inference 및 데이터 처리를 위한 함수 관련 모듈

# arguments
main.py [--train] [--eval] [--target_model rgrs, clsf, joint]

# Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-00330, Development of AI Technology for Early Screening of Child/Child Autism Spectrum Disorders based on Cognition of the Psychological Behavior and Response).

# Contributors
김홍국 (hongkook@gist.ac.kr, GIST, 교수)

이정혁 (ljh0412@gist.ac.kr, GIST, 박사과정)

이건우 (geonwoo0801@gist.ac.kr, GIST, 박사과정)

전지민 (jiminbot20@gm.gist.ac.kr, GIST, 박사과정)

박동건  (dongkeon@gist.ac.kr, GIST, 통합과정)
