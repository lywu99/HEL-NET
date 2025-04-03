# HEL-NET
Official Pytorch Implementation of : Heterogeneous Ensemble Learning for Comprehensive Diabetic Retinopathy Multi-Lesion Segmentation via Mamba-UNet
# Requirements
  Inatall pytorch =1.9.3, torchvision, numpy, opencv-python, tqdm, protobuf,optparse
  Install Mamba: pip install causal-conv1d=1.0.0, pip install mamba-ssm

# Usage
## 1. Data Preparation
   IDRiD and DDR datasets can be downloaded in the following links:
  1. IDRiD Dataset-[Link](https://idrid.grand-challenge.org/)
  2. DDR Dataset-[Link](https://github.com/nkicsl/DDR-dataset)

Make your data directory like this below
```language
├── data
│   ├── OriginalImages
       ├── TraningSet
           ├── Haemorrhages
           ├── SoftExudates
           ├── Microaneurysms
           ├── HardExudates
       ├── TestingSet
          '''
    ├── Groundtruths
       ├── TraningSet
          '''
       ├── TestingSet
          '''
```
2. Data preprcessing, please run preprocess.py  to get the fundus image.


# 2.Training
Run:
```language
   python train.py
```
The model point are saved in ./logs/results/model_name/weights/model_AUPR.pth.tar


# 3.Testing
Run:
```language
   python eval.py 
```

# 4. Citation and contact
