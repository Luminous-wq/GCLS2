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
     


def train(curs, G: nx.Graph, GH: nx.Graph, new_labels, new_features):
    device = 'cuda'
    adjacency_matrices_G = []
    adjacency_matrices_GH = []
    labels_all = []
    feature_all = []
    losslist = []
    acclist = []
    gcntimelist =[]
    cltimelist = []

    #model = GAT(1657,256,8,0.6,0.2,8).to(device)
    model_cl = GSCL(1657,256,8,3,8,1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_cl.parameters(),lr=0.001)
    max_acc =0

    for i in range(len(curs)-1):
        losslist_tmp = []
        acclist_tmp = []
        gcntimelist_tmp = []
        cltimelist_tmp = []
        sub_list = list(curs[i])
        # print(sub_list)
        subG = G.subgraph(sub_list)
        subGH = GH.subgraph(sub_list)
        feature1 = new_features[sub_list].to(device)
        label1 = new_labels[sub_list].to(device)


        adj_matrix_G = np.array(nx.adjacency_matrix(subG).todense())
        adj_matrix_GH = np.array(nx.adjacency_matrix(subGH).todense())
        edge_index = torch.tensor(adj_matrix_G).to(device)
        
        cl_epoch_num = 50
        gcn_epoch_num = 200

        cl_start = time.time()

        for epoch in range(cl_epoch_num):
            optimizer.zero_grad()
            loss = 0
            loss = model_cl(edge_index,edge_index,feature1,feature1)
            loss.backward()
            optimizer.step()
            
        cl_end = time.time()

        gcn_start = time.time()
        max_acc = 0
        nmi = 0
        f1 = 0
        for epoch in range(gcn_epoch_num):

            optimizer.zero_grad()
            
            pred = model_cl.get_embedding(edge_index,feature1)
            loss = criterion(pred, label1)
            correct = (pred.argmax(dim=1) == label1).sum().item()
            cnt = len(feature1)
            
            loss.backward()
            optimizer.step()

            losslist_tmp.append(loss.item())
            acclist_tmp.append(correct/cnt)
        
            if(correct/cnt>max_acc):
                max_acc = correct/cnt
            print('epoch',epoch)
            print('loss',loss.item())
            print(correct)
            print(cnt)
            print(correct/cnt)
            print()
        gcn_end = time.time()
        losslist.append(losslist_tmp)
        acclist.append(acclist_tmp)
        gcntimelist.append(gcn_end-gcn_start)
        cltimelist.append(cl_end-cl_start)

        
    sub_list = list(curs[-1])

    subG = G.subgraph(sub_list)
    subGH = GH.subgraph(sub_list)


    adj_matrix_G = np.array(nx.adjacency_matrix(subG).todense())
    adj_matrix_GH = np.array(nx.adjacency_matrix(subGH).todense())
    edge_index = torch.tensor(adj_matrix_G).to(device)

    feature1 = new_features[sub_list].to(device)
    label1 = new_labels[sub_list].to(device)
    pred = model_cl.get_embedding(edge_index,feature1)

    nmi = metrics.normalized_mutual_info_score(pred.argmax(dim=1).cpu().detach().numpy(),label1.cpu().detach().numpy())
    f1 = f1_score(pred.argmax(dim=1).cpu().detach().numpy(),label1.cpu().detach().numpy(), average='weighted')

    correct = (pred.argmax(dim=1) == label1).sum().item()
    cnt = len(feature1)

    print('final ')
    print('max_acc',max_acc)
    print(correct/cnt)
    print(nmi)
    print(f1)

    print(gcntimelist)
    print(cltimelist)

    
    return adjacency_matrices_G, adjacency_matrices_GH, labels_all, feature_all