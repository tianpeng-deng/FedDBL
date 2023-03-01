# FedDBL: Communication and Data Efficient Federated Deep-Broad Learning for Histopathological Tissue Classification
![outline](FedDBL)

## Introduction
The implementation of:

[FedDBL: Communication and Data Efficient Federated Deep-Broad Learning for Histopathological Tissue Classification](https://arxiv.org/abs/2302.12662)

## Abstract
Histopathological tissue classification is a fundamental task in computational pathology. Deep learning-based models have achieved superior performance but centralized training with data centralization suffers from the privacy leakage problem. Federated learning (FL) can safeguard privacy by keeping training samples locally, but existing FL-based frameworks require a large number of well-annotated training samples and numerous rounds of communication which hinder their practicability in the real-world clinical scenario. In this paper, we propose a universal and lightweight federated learning framework, named Federated Deep-Broad Learning (FedDBL), to achieve superior classification performance with limited training samples and only one-round communication. By simply associating a pre-trained deep learning feature extractor, a fast and lightweight broad learning inference system and a classical federated aggregation approach, FedDBL can dramatically reduce data dependency and improve communication efficiency. Five-fold cross-validation demonstrates that FedDBL greatly outperforms the competitors with only one-round communication and limited training samples, while it even achieves comparable performance with the ones under multiple-round communications. Furthermore, due to the lightweight design and one-round communication, FedDBL reduces the communication burden from 4.6GB to only 276.5KB per client using the ResNet-50 backbone at 50-round training. Since no data or deep model sharing across different clients, the privacy issue is well-solved and the model security is guaranteed with no model inversion attack risk.

## Requirements
- CUDA
- 1×GPU
- 2×CPU
- Python 3.9.12
- numpy==1.22.4
- pytorch==1.12.0
- torchvision==0.13.0
- scikit-learn==1.1.1


## Usage
### Installation
- Download the repository.
```
git clone https://github.com/tianpeng-deng/FedDBL.git
```
- Install python dependencies.

### Data and Backbone Preparation
- We provide an example to train a FedDBL on Multi-center CRC and BCSS, where the dataset can be downloaded from [Multi-center CRC](https://doi.org/10.1016/j.ebiom.2020.103054) and [BCSS](https://bcsegmentation.grand-challenge.org/). 
- All backbones (pre-trained models) are stored in *load* folder. The CTransPath backbone is provided [here](https://github.com/Xiyue-Wang/TransPath). 
- The architecture of dataset is provided in folder "dataset" and also illustrated below (more details can be seen in /dataset/CRC/partitions.txt):
```
    |-- dataset
        |-- CRC
        |    |-- 100
        |    |    |-- Fold 1
        |    |    |    |-- Client 1
        |    |    |    |    |-- Train 
        |    |    |    |    |-- Valid
        |    |    |    |-- Client 2
        |    |    |    |-- Client 3
        |    |    |    |-- Client 4
        |    |    |    |-- Centralized
        |    |    |-- Fold 2
        |    |-- 070
        |    |    |-- Fold 1
        |    |    |    |-- Client 1
        |    |    |    |    |-- Train 
        |    |    |    |-- Client 2
        |    |    |    |-- Client 3
        |    |    |    |-- Client 4
        |    |    |    |-- Centralized
        |    |    |-- Fold 2
        |    |-- 050
        |-- BC
```

There is only one "Valid" for each Fold and every partitions use the same "Valid" for fair comparison.

### FedDBL
-  To use this example code, you must prepare different proportions of dataset and store them in **datset** folder. And you can train and test the FedDBL by simply using the command:

```
python main.py
```


To run on BCSS, you need to change the weights because there are three users in BCSS but four in Multi-center CRC.
## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@misc{https://doi.org/10.48550/arxiv.2302.12662,
  doi = {10.48550/ARXIV.2302.12662},
  url = {https://arxiv.org/abs/2302.12662},
  author = {Deng, Tianpeng and Huang, Yanqi and Shi, Zhenwei and Lin, Jiatai and Dou, Qi and Zhao, Ke and Liu, Fang-Fang and Jia, Yu-Mian and Wang, Jin and Zhao, Bingchao and Liang, Changhong and Liu, Zaiyi and Guo, Xiao-jing and Han, Guoqiang and Chen, Xin and Han, Chu},
  title = {FedDBL: Communication and Data Efficient Federated Deep-Broad Learning for Histopathological Tissue Classification},
  publisher = {arXiv},
  year = {2023}
}

```
