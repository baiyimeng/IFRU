# IFRU
[![arXiv](https://img.shields.io/badge/arXiv-2307.02147-red.svg)](https://arxiv.org/abs/2307.02147)

This is the PyTorch implementation of our paper published in ACM Transactions on Recommender Systems (ACM TORS):

> **[Recommendation Unlearning via Influence Function](https://arxiv.org/abs/2307.02147)**  
> Yang Zhang, Zhiyu Hu, Yimeng Bai, Jiancan Wu, Qifan Wang, Fuli Feng

## Usage

### Data
Download the original datasets via the link provided in `Data/download.txt`, and preprocess them using the `_data_process.py` script.

### Training & Evaluation
All run scripts used in the paper are named according to the method, backbone, dataset, and other configurations.  
For example, `eraser_mf_amazon.py` corresponds to the **RecEraser** method with **MF** as the backbone, applied on the **Amazon** dataset:

```
python eraser_mf_amazon.py
```

## Citation
```
@article{IFRU,
author = {Zhang, Yang and Hu, Zhiyu and Bai, Yimeng and Wu, Jiancan and Wang, Qifan and Feng, Fuli},
title = {Recommendation Unlearning via Influence Function},
year = {2024},
issue_date = {June 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {3},
number = {2},
url = {https://doi.org/10.1145/3701763},
doi = {10.1145/3701763},
journal = {ACM Trans. Recomm. Syst.},
month = dec,
articleno = {22},
numpages = {23},
keywords = {Recommender system, recommendation unlearning, privacy, influence function}
}
```
