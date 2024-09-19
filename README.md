# Satellite-Image-Building-Area-Segmentation
데이콘 SW대학 AI공동경진대회 잉카인터넷(특별상) 21등 수상 코드

![스크린샷 2024-09-19 174832](https://github.com/user-attachments/assets/06d16818-e0b6-438e-bada-353425268883)

<br>

# 대회 설명
- 주제 : 위성 이미지 건물 영역 분할
(Satellite Image Building Area Segmentation)

- 문제 : 위성 이미지의 건물 영역 분할(Image Segmentation)을 수행하는 AI모델을 개발

<br>

# 팀원

| 김의진 | 장희진 | 박지원 | 정승민 |
| :---: | :---: | :---: | :---: |
| <img src="https://avatars.githubusercontent.com/u/94896197?v=4" width=300>  | <img src="https://avatars.githubusercontent.com/u/105128163?v=4" width=300> | <img src="https://github.com/user-attachments/assets/2f1cd234-d276-4888-bb16-a1a1dc821950" width=300> | <img src="https://avatars.githubusercontent.com/u/105360496?v=4" width=300>

<br>

# 환경

- 우분투 22.04

- NVIDIA-SMI 535.54.03   

- CUDA Version: 11.8

- setting reference : https://vividian.net/2022/11/1111 

<br>

# 라이브러리

 - environment/requirements.txt 와 environment/conda-packages.txt 참고

<br>
 
# 모델

- 최고 weight : checkpoint_efun7_fr_bc_cl.pt

- 사용한 사전 학습 모델 : ImageNet

- Efficient U-Net template : https://github.com/zhoudaxia233/EfficientUnet-PyTorch 

- Pre-Trained Encoder: EfficientNet with ImageNet Dataset

- train_background.sh 를 통해 백그라운드 학습 가능

<br>

# 데이터 위치

- 기본 값 DBDIR = "/hdd_data/datasets/aicon" 

- --db_dir로 디렉토리 변경 가능

- ex) --db_dir /data

<br>

# 가상환경

- conda create --name aicon python=3.11.4 -y

- conda activate aicon

- conda install -y opencv

- conda install -y numpy

- conda install -y pandas

- conda install -y matplotlib

- conda install -y tqdm

- conda install -y scikit-learn

- conda install -y -c conda-forge albumentations

- pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

- pip install efficientunet-pytorch

- environment/environment.sh 로 가상환경 자동 생성 가능

# 실행

- 최고기록 : python3 -u train.py -b 32 -lr 2e-04 -aug -aug_fr -aug_bc -aug_cl -nep 20 -pf efun7_fr_bc_cl_$today

# 파일 설명

- ./environment/conda-packages-Nopy.txt : py가 없는 대회 conda 세팅 (참고용)

- ./environment/conda-packages.txt : py가 있는 대회 conda 세팅 (참고용)

- ./environment/requirements.txt : pip 라이브러리 list (참고용)

- ./environment/environment.sh : 자동 세팅 파일

- ./log : log 파일 기록 디렉토리

- ./modules/dataset.py : 데이터 전처리 .py

- ./modules/decode.py : rle encoder 와 decoder

- ./modules/dice_eval.py : dice score 계산 .py

- ./modules/early_stop.py : early_stop 함수 .py

- ./train_background.sh : train.py 자동 백그라운드 실행 nohup

- ./train_multi_GPU.py : 멀티 GPU 기능이 있는 .py 파일

- ./train.pt : 메인 train 파일 
