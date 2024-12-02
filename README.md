## GCLS2: Towards Efficient Community Detection Using Graph Contrastive Learning with Structure Semantics

This is the official code-repo for "**GCLS2: Towards Efficient Community Detection Using Graph Contrastive Learning with Structure Semantics**". [[Paper]](https://arxiv.org/abs/2410.11273)

If you find this work useful in your method, you can cite the paper as below:

```
@article{wen2024gcls,
  title={GCLS $\^{} 2$: Towards Efficient Community Detection using Graph Contrastive Learning with Structure Semantics},
  author={Wen, Qi and Zhang, Yiyang and Ye, Yutong and Zhou, Yingbo and Zhang, Nan and Lian, Xiang and Chen, Mingsong},
  journal={arXiv preprint arXiv:2410.11273},
  year={2024}
}
```

### Code-Repo Introduction

This code-repo contains the following contents:

```
.
├─README.md
│
├─datasets				--- This folder contains the original mini-datasets. 
│  ├─ citeseer
│  ├─ cora
│  ├─ cs
│  ├─ email-eu
│  ├─ photo
│  └─ pubmed
│
├─large-training			--- This folder contains codes of GCLS2 for large-datasets.		
│      ├─ main.py			(run GCLS2)
│      ├─ model.py			(all model and loss calulate)
│      ├─ preproc.py			(partitioning methods: HGP, LDG, METIS and Hash)
│      └─ train.py			(training processing)
│
├─logs					--- This folder contains running logs for training.			
│      └─ cora.log
│
├─mini-training				--- This folder contains codes of GCLS2 for mini-datasets.	
│      ├─ argparses.py			(argparses)
│      ├─ main.py			(run GCLS2)
│      ├─ model.py			(all model and loss calulate)
│      └─ read_datasets.py		(read and deal the origin mini-datasets)
│
└─ptModel				--- This folder contains pre-trained and test models for reproducibility.	
       ├─ best_model_cl_cora.pt
       └─ best_model_cora.pt

```

## Required environment

1. python 3.8 or above
2. torch 2.1.0 or above
3. torch_geometric (Please note the version matching [[PyTorch]](https://pytorch.org/get-started/previous-versions/).)
4. networkx 3.1
5. sklearn
6. pymetis

## Running way

```
For mini-datasets:
a): change the read function in main.py
b): change the training parameters in args and main.py
c): python main.py

For lagre-datasets:
a): change the *.gml path (datapath and datapath_H) in main.py
b): change the dataset class parameter in preproc.py
c): python main.py
```

## About Datasets

All datasets are obtained by [[pytorch-geometric]](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html) and [[SNAP]](https://snap.stanford.edu/data/#socnets). 

TW and AMA are over 20G, we can't upload large datasets due to the rule limitations, but they can be obtained from [[pytorch-geometric]](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html).



#### 