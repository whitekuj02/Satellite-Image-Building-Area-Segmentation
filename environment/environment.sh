#!/bin/bash

# Conda 초기화
eval "$(conda shell.bash hook)"

# 새로운 환경 생성
conda create --name aicon python=3.6 -y

# 환경 활성화
conda activate aicon

# 필요한 패키지들 설치
conda install -n aicon -y opencv
conda install -n aicon -y numpy
conda install -n aicon -y pandas
conda install -n aicon -y matplotlib
conda install -n aicon -y tqdm
conda install -n aicon -y scikit-learn
conda install -n aicon -y -c conda-forge albumentations
conda install -n aicon -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# pip로 패키지 설치
pip install efficientunet-pytorch
