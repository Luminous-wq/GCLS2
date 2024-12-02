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
     
class Partitioner:
    def __init__(self, adj_matrix: List[List], adj_matrix_H: List[List]):
        """
        图分区的构造函数
        :param adj_matrix: 邻接矩阵
        """
        self.adj_matrix = adj_matrix
        self.adj_matrix_H = adj_matrix_H
        self.num_vertices = len(adj_matrix)
        self.num_edges = sum(len(row) for row in adj_matrix)
        self.num_edges_H = sum(len(row) for row in adj_matrix_H)
        self.part_results = []
        self.cur_parts = []

    def execute(self, part_nums: int):
        """
        图分割算法
        :param part_nums: 分区个数
        """
        raise NotImplementedError("This method was implemented by subclasses.")

    def evaluate(self):
        """
        评估图分割算法
        """
        raise NotImplementedError("This method was implemented by subclasses.")

    def get_results(self) -> List:
        """
        返回分区结果
        :return: 分区结果
        """
        return self.part_results, self.cur_parts

    def get_num_vertices(self) -> int:
        """
        获取顶点总数
        :return: 顶点总数
        """
        return self.num_vertices

    def get_num_edges(self) -> int:
        """
        获取图的边的总数
        :return: 边的总数
        """
        return self.num_edges
    
    def get_num_edges(self) -> int:
        """
        获取高阶图的边的总数
        :return: 高阶图边的总数
        """
        return self.num_edges_H

class LDGPartitioner(Partitioner):
    # def __init__(self, adj_matrix: List[List[int]]):
    def __init__(self, adj_matrix: List[List], adj_matrix_H: List[List]):
        super().__init__(adj_matrix, adj_matrix_H)
        

    def execute(self, part_nums: int):
        """
        图分割算法
        :param part_nums: 分区个数
        """
        # 初始化
        order = list(range(self.num_vertices))  # 节点 ID 集合
        random.shuffle(order)  # 随机打乱节点顺序
        self.cur_parts = [set() for _ in range(part_nums)]  # 每个分区的节点集合
        self.part_results = [-1] * self.num_vertices  # 初始化分区结果

        # 初始化每个分区的第一个节点
        for i in range(part_nums):
            self.cur_parts[i].add(order[i])
            self.part_results[order[i]] = i
    
        expectant = self.num_vertices / part_nums
        # 遍历剩余节点
        for vertex in order[part_nums:]:
            scores = []
            if len(self.adj_matrix_H[vertex])==0:
                for j in range(part_nums):
                 
                    # weight = expectant / (cur_size+1) # 权重
                    neighbors = self._intersect(vertex, self.cur_parts[j])  # 邻居数量
                    scores.append(neighbors)
            else:
                for j in range(part_nums):
                    cur_size = len(self.cur_parts[j])
                    weight = 1 - (cur_size / expectant)  
                    # 权重
                    neighbors = self._intersect(vertex, self.cur_parts[j])  # 邻居数量
                    scores.append(neighbors * weight)

            # print(scores)
            # 将节点分配到得分最高的分区
            #print(scores)
            max_index = scores.index(max(scores))
            self.cur_parts[max_index].add(vertex)
            self.part_results[vertex] = max_index

    def _intersect(self, vertex, part: Set) -> int:
        """
        计算一个节点与某个分区之间的邻居交集数量
        :param vertex: 节点 ID
        :param part: 分区中的节点集合
        :return: 邻居交集数量
        """
        neighbors = set(self.adj_matrix[vertex])  # 当前节点的邻居集合
        # print(neighbors, part)
        return len(neighbors & part)
    
    def _intersect_H(self, vertex, part: Set) -> int:
        """
        计算一个节点与某个分区之间的高阶邻居交集数量
        :param vertex: 节点 ID
        :param part: 分区中的节点集合
        :return: 高阶邻居交集数量
        """
        neighbors = set(self.adj_matrix_H[vertex])  # 当前节点的高阶邻居集合
        return len(neighbors & part)
class HGPPartitioner(Partitioner):
    # def __init__(self, adj_matrix: List[List[int]]):
    def __init__(self, adj_matrix: List[List], adj_matrix_H: List[List]):
        super().__init__(adj_matrix, adj_matrix_H)
        

    def execute(self, part_nums: int):
        """
        图分割算法
        :param part_nums: 分区个数
        """
        # 初始化
        order = list(range(self.num_vertices))  # 节点 ID 集合
        random.shuffle(order)  # 随机打乱节点顺序
        self.cur_parts = [set() for _ in range(part_nums)]  # 每个分区的节点集合
        self.part_results = [-1] * self.num_vertices  # 初始化分区结果

        # 初始化每个分区的第一个节点
        for i in range(part_nums):
            self.cur_parts[i].add(order[i])
            self.part_results[order[i]] = i
   
        # 每个分区的期望大小
        expectant = self.num_vertices / part_nums
        # 遍历剩余节点
        for vertex in order[part_nums:]:
            scores = []
            # 孤立顶点
            if len(self.adj_matrix_H[vertex])==0:
                for j in range(part_nums):
                    # 负载均衡
                    if len(self.cur_parts[j]) >= expectant:
                        scores.append(-100)
                    else:
                        neighbors = self._intersect(vertex, self.cur_parts[j])
                        scores.append(neighbors)
            else:
                for j in range(part_nums):
                    if len(self.cur_parts[j]) >= expectant: 
                        scores.append(-100)
                    else:
                        cur_size = len(self.cur_parts[j])
                        # weight = 1 - (cur_size / expectant)  
                        weight = expectant / (cur_size+1) # 权重
                        neighbors = self._intersect_H(vertex, self.cur_parts[j]) # 邻居数量
                        scores.append(neighbors * weight)

            # print(scores)
            # 将节点分配到得分最高的分区
            #print(scores)
            max_index = scores.index(max(scores))
            self.cur_parts[max_index].add(vertex)
            self.part_results[vertex] = max_index


    def _intersect(self, vertex, part: Set) -> int:
        """
        计算一个节点与某个分区之间的邻居交集数量
        :param vertex: 节点 ID
        :param part: 分区中的节点集合
        :return: 邻居交集数量
        """
        neighbors = set(self.adj_matrix[vertex])  # 当前节点的邻居集合
        # print(neighbors, part)
        return len(neighbors & part)
    
    def _intersect_H(self, vertex, part: Set) -> int:
        """
        计算一个节点与某个分区之间的高阶邻居交集数量
        :param vertex: 节点 ID
        :param part: 分区中的节点集合
        :return: 高阶邻居交集数量
        """
        neighbors = set(self.adj_matrix_H[vertex])  # 当前节点的高阶邻居集合
        return len(neighbors & part)

def create_folder(folder_name):
    base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_name, exist_ok=True)
    return True

def read_amap():
    dataset = AmazonProducts(root='./dataset/amazonP')
    data = dataset._data
    key_num = []

    G = nx.Graph()
    temp_data = data.edge_index[0]
    temp_data_2 = data.edge_index[1]
    print(temp_data)
    print(temp_data_2)
    l = len(data.edge_index[0])
    count = 0
    for i in range(l):
        
        v1 = temp_data[i]
        v2 = temp_data_2[i]
        # print(type(v1))
        G.add_edge(int(v1), int(v2))
    return G
def read_tweibo():
    dataset = AttributedGraphDataset(root='/home/mastlab/wq/S3AND/dataset',name='tweibo')
    data = dataset._data
    key_num = []

    G = nx.Graph()

    temp_data = data.edge_index[0]
    temp_data_2 = data.edge_index[1]
    print(temp_data)
    print(temp_data_2)
    l = len(data.edge_index[0])
    count = 0
    for i in range(l):
        
        v1 = temp_data[i]
        v2 = temp_data_2[i]
        G.add_edge(int(v1), int(v2))
    return G

def random_node_partition(graph_nodes, k):
    """
    随机哈希的节点分割
    Args:
        graph_nodes (list): 图中的节点列表。
        k (int): 分割子图的数量。
    Returns:
        partitions (dict): 子图分割结果，每个子图包含其节点列表。
    """
    partitions = [[] for i in range(k)]  # 初始化k个子图
    for node in graph_nodes:
        partition_id = hash(node) % k  # 使用哈希函数并取模
        partitions[partition_id].append(node)  # 将节点加入对应子图
    return partitions

def pymetis_partition(G, num_parts):
    """
    使用 pymetis 对图进行分割
    :param G: NetworkX 图
    :param num_parts: 分区数
    :return: 每个节点所属的分区列表
    """
    # 构造 METIS 所需的图数据
    adjacency_list = [list(G.neighbors(node)) for node in G.nodes()]
    # 使用 pymetis 分割图
    _, parts = pymetis.part_graph(num_parts, adjacency=adjacency_list)
    partitions = parts
    partition_sizes = [0] * num_parts
    for part in partitions:
        partition_sizes[part] += 1
    curs = []
    for part_id in range(num_parts):
    # 提取属于当前分区的节点
        nodes_in_part = [node for node, part in enumerate(partitions) if part == part_id]
        curs.append(nodes_in_part)
    return curs

def split(datapath, datapath_H, p, dataname): #p是分块个数   dataname 保存文件名
    """
    主程序，测试 HGPPartitioner 的功能
    """

    device = 'cuda'
    seed = 66
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    t1 = time.time()
    G1 = read_tweibo()

    print('load')

    # 处理id为字符或其他，转为数字
    mapping = {}
    for i, node in enumerate(G1.nodes()):
        mapping[node] = i
    # print(mapping)
    G = nx.relabel_nodes(G1, mapping)

    dataset = AttributedGraphDataset(root='/home/mastlab/wq/S3AND/dataset',name='tweibo')
    data = dataset[0]
    labels = data['y']
    # print(labels[0:10])
    node_feature=data['x']
    node_num = data.num_nodes
    #new_labels = torch.empty((node_num,107), dtype=labels.dtype)
    new_labels = torch.empty(node_num, dtype=labels.dtype)
    new_features = torch.empty((node_num, len(node_feature[0])), dtype=node_feature.dtype)
    for old_index, new_index in mapping.items():
        old_index = int(old_index)  # Convert keys to int
        if old_index < len(labels) and old_index < len(node_feature) and new_index < node_num:
            new_labels[new_index] = labels[old_index]
            new_features[new_index] = node_feature[old_index]
        else:
            # print(f"Skipping invalid mapping: old_index={old_index}, new_index={new_index}")
            continue
    
    print(new_features, new_labels)
    print(new_features.size())
    print(new_labels.size())

    t2 = time.time()
    print("read gml and high gml need time:{}".format(t2-t1))

    adj_list = [list(G.neighbors(node)) for node in G.nodes()]

    print(len(adj_list))#, len(adj_list_H))
    t3 = time.time()
    print("get adj need time:{}".format(t3-t2))


    partitioner = HGPPartitioner(adj_list, adj_list)
    partitioner.execute(p)  # 将图分成 p 个部分
    # # 获取分区结果
    ans, curs = partitioner.get_results()

    print("每个区有: ")
    with open("partition_results_"+dataname+"_cur.txt", "w") as f:
        for t,cur in enumerate(curs):
            print("{}, {}".format(t, len(cur)))
            f.write(f"{t} {cur}\n")

    print("分区结果已保存到 partition_results_"+dataname+"_cur.txt")
    t5 = time.time()
    print("get all need time:{}".format(t5-t1))

    # ans_A, ans_H = txt2subgraph("partition_results_"+dataname+"_cur.txt", G, GH)
    adjacency_matrices_G, adjacency_matrices_GH, labels_all, feature_all = train(curs, G, G, new_labels=new_labels, new_features=new_features)#cur是节点列表

    print("分区label大小: {}".format(len(labels_all)))
    
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