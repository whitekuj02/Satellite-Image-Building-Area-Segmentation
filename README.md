# Satellite-Image-Building-Area-Segmentation
# 환경

우분투 22.04

NVIDIA-SMI 535.54.03   

CUDA Version: 11.8..?

# pip 라이브러리 버전

torch==2.0.0+cu118
torchvision==0.15.1+cu118            
...

albumentations            1.0.3              pyhd8ed1ab_0    conda-forge
blas                      1.0                         mkl  
bzip2                     1.0.8                h7b6447c_0  
ca-certificates           2023.7.22            hbcca054_0    conda-forge
cairo                     1.16.0               hb05425b_5  
certifi                   2021.5.30        py36h5fab9bb_0    conda-forge
cloudpickle               2.2.1              pyhd8ed1ab_0    conda-forge
colorama                  0.4.4              pyhd3eb1b0_0  
cuda-cudart               11.8.89                       0    nvidia
cuda-cupti                11.8.87                       0    nvidia
cuda-libraries            11.8.0                        0    nvidia
cuda-nvrtc                11.8.89                       0    nvidia
cuda-nvtx                 11.8.86                       0    nvidia
cuda-runtime              11.8.0                        0    nvidia
cudatoolkit               11.0.221             h6bb024c_0    nvidia
cycler                    0.11.0             pyhd3eb1b0_0  
cytoolz                   0.11.0           py36h8f6f2f9_3    conda-forge
dask-core                 2021.3.0           pyhd8ed1ab_0    conda-forge
dataclasses               0.8                pyh4f3eec9_6  
dbus                      1.13.18              hb2f20db_0  
efficientunet-pytorch     0.0.6                    pypi_0    pypi
expat                     2.4.9                h6a678d5_0  
ffmpeg                    4.0                  hcdf2ecd_0  
fontconfig                2.14.1               h52c9d5c_1  
freeglut                  3.0.0                hf484d3e_5  
freetype                  2.12.1               h4a9f257_0  
future                    0.18.3                   pypi_0    pypi
geos                      3.9.1                h9c3ff4c_2    conda-forge
glib                      2.69.1               h4ff587b_1  
graphite2                 1.3.14               h295c915_1  
gst-plugins-base          1.14.1               h6a678d5_1  
gstreamer                 1.14.1               h5eee18b_1  
harfbuzz                  1.8.8                hffaf4a1_0  
hdf5                      1.10.2               hba1933b_1  
icu                       58.2                 he6710b0_3  
imageio                   2.13.1             pyhd8ed1ab_0    conda-forge
imgaug                    0.4.0              pyhd8ed1ab_1    conda-forge
intel-openmp              2022.1.0          h9e868ea_3769  
jasper                    2.0.14               hd8c5072_2  
joblib                    1.0.1              pyhd3eb1b0_0  
jpeg                      9e                   h5eee18b_1  
kiwisolver                1.3.1            py36h2531618_0  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libcublas                 11.11.3.6                     0    nvidia
libcufft                  10.9.0.58                     0    nvidia
libcufile                 1.7.0.149                     0    nvidia
libcurand                 10.3.3.53                     0    nvidia
libcusolver               11.4.1.48                     0    nvidia
libcusparse               11.7.5.86                     0    nvidia
libdeflate                1.17                 h5eee18b_0  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            7.5.0               ha8ba4b0_17  
libgfortran4              7.5.0               ha8ba4b0_17  
libglu                    9.0.0                hf484d3e_1  
libgomp                   11.2.0               h1234567_1  
libnpp                    11.8.0.86                     0    nvidia
libnvjpeg                 11.9.0.86                     0    nvidia
libopencv                 3.4.2                hb342d67_1  
libopus                   1.3.1                h7b6447c_0  
libpng                    1.6.39               h5eee18b_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.5.0                h6a678d5_2  
libuuid                   1.41.5               h5eee18b_0  
libuv                     1.44.2               h5eee18b_0  
libvpx                    1.7.0                h439df22_0  
libwebp-base              1.2.4                h5eee18b_1  
libxcb                    1.15                 h7f8727e_0  
libxml2                   2.9.14               h74e7548_0  
lz4-c                     1.9.4                h6a678d5_0  
matplotlib                3.3.4            py36h06a4308_0  
matplotlib-base           3.3.4            py36h62a2d02_0  
mkl                       2020.2                      256  
mkl-service               2.3.0            py36he8ac12f_0  
mkl_fft                   1.3.0            py36h54f3939_0  
mkl_random                1.1.1            py36h0573a6f_0  
ncurses                   6.4                  h6a678d5_0  
networkx                  2.7                pyhd8ed1ab_0    conda-forge
ninja                     1.10.2               h06a4308_5  
ninja-base                1.10.2               hd09550d_5  
numpy                     1.19.2           py36h54aff64_0  
numpy-base                1.19.2           py36hfa32c7d_0  
olefile                   0.46                     py36_0  
opencv                    3.4.2            py36h6fd60c2_1  
openjpeg                  2.4.0                h3ad879b_0  
openssl                   1.1.1o               h166bdaf_0    conda-forge
pandas                    1.1.5            py36ha9443f7_0  
pcre                      8.45                 h295c915_0  
pillow                    8.3.1            py36h2c7a002_0  
pip                       21.2.2           py36h06a4308_0  
pixman                    0.40.0               h7f8727e_1  
py-opencv                 3.4.2            py36hb342d67_1  
pyparsing                 3.0.4              pyhd3eb1b0_0  
pyqt                      5.9.2            py36h05f1152_2  
python                    3.6.13               h12debd9_1  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python_abi                3.6                     2_cp36m    conda-forge
pytorch                   1.7.0           py3.6_cuda11.0.221_cudnn8.0.3_0    pytorch
pytorch-cuda              11.8                 h7e8668a_5    pytorch
pytz                      2021.3             pyhd3eb1b0_0  
pywavelets                1.1.1            py36he33b4a0_3    conda-forge
pyyaml                    5.4.1            py36h8f6f2f9_1    conda-forge
qt                        5.9.7                h5867ecd_1  
readline                  8.2                  h5eee18b_0  
scikit-image              0.16.2           py36hb3f55d8_0    conda-forge
scikit-learn              0.24.2           py36ha9443f7_0  
scipy                     1.5.2            py36h0b6359f_0  
setuptools                58.0.4           py36h06a4308_0  
shapely                   1.7.1            py36hff28ebb_5    conda-forge
sip                       4.19.8           py36hf484d3e_0  
six                       1.16.0             pyhd3eb1b0_1  
sqlite                    3.41.2               h5eee18b_0  
threadpoolctl             2.2.0              pyh0d69192_0  
tk                        8.6.12               h1ccaba5_0  
toolz                     0.12.0             pyhd8ed1ab_0    conda-forge
torchvision               0.8.1                py36_cu110    pytorch
tornado                   6.1              py36h27cfd23_0  
tqdm                      4.63.0             pyhd3eb1b0_0  
typing_extensions         4.1.1              pyh06a4308_0  
wheel                     0.37.1             pyhd3eb1b0_0  
xz                        5.4.2                h5eee18b_0  
yaml                      0.2.5                h7f98852_2    conda-forge
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.5                hc292b87_0  


자세한 버전은 ./environment/requirements.txt 와 ./environment/conda-packages.txt 참고

 
# 모델

최고 weight : checkpoint_efun7_fr_bc_cl.pt

사용한 사전 학습 모델 : ImageNet

train_background.sh 를 통해 백그라운드 학습 가능

# 데이터 위치

기본 값 DBDIR = "/hdd_data/datasets/aicon" 

--db_dir로 디렉토리 변경 가능

--db_dir /data

# 가상환경

conda create --name aicon python=3.6

conda activate aicon

conda install -y opencv
conda install -y numpy
conda install -y pandas
conda install -y matplotlib
conda install -y tqdm
conda install -y scikit-learn
conda install -y -c conda-forge albumentations
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

pip install efficientunet-pytorch

./environment/environment.sh 로 가상환경 자동 생성 가능

# 실행
최고기록

python3 -u train.py -b 32 -lr 2e-04 -aug -aug_fr -aug_bc -aug_cl -nep 20 -pf efun7_fr_bc_cl_$today
