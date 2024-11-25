import os
import random
from typing import List, Set
import networkx as nx
import time
import pymetis
import numpy as np
import torch
from torch_geometric.datasets import AmazonProducts, AttributedGraphDataset
from model import GSCL,GAT
from sklearn import metrics
from sklearn.metrics import f1_score
from preproc import split


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    # datapath = "../dataset/G-123182-778561.gml"
    # datapath = "../dataset/G.gml"
    # datapath = "../dataset/G-1944589-50133382.gml"
    # datapath = "../dataset/cora/cora_orig.gml"
    #datapath = "./dataset/TW/G-1944589-50133382.gml"
    datapath = "/home/mastlab/wq/S3AND/dataset/precompute/real/tweibo/G-1944589-50133382.gml"
    # datapath = "/home/yons/wq/S3AND/dataset/precompute/real/ama/G-1569960-132954714.gml"

    datapath_H = "/home/mastlab/wq/S3AND/dataset/precompute/real/tweibo/G-1944589-50133382.gml"
    # datapath_H = "../dataset/cora/cora_high.gml"
    # datapath_H = "/home/yons/wq/S3AND/dataset/precompute/real/ama/G-1569960-132954714.gml"
    p = 256# 3, 16, 64
    dataname = "tweibo" # facebook, yago3, tw
    split(datapath=datapath, datapath_H=datapath_H, p=p, dataname=dataname)
    
    



