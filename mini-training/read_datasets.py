import networkx as nx
import numpy as np
import torch
import graph_tool as gt
import graph_tool.topology as gt_topology
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid,KarateClub,Amazon,Coauthor


def read_photo():
    print("photo")
    dataset=Amazon(root='./tmp/photo',name='Photo')
    data=dataset[0]
    node_feature=data['x']
    setattr(data, 'edges', data.edge_stores[0]['edge_index'].t())
    edges = [list([pair[0].item(),pair[1].item()]) for pair in data.edges]
    data_graph=nx.Graph()
    data_graph.add_edges_from(edges)

    edge_support = {}

    # 计算每条边出现在多少个三角形中
    for u, v in data_graph.edges():
        # 获取共同邻居（即形成三角形的第三个节点）
        common_neighbors = list(nx.common_neighbors(data_graph, u, v))
        support = len(common_neighbors)
        edge_support[(u, v)] = support
    for u, v in data_graph.edges():
        data_graph[u][v]['support'] = edge_support[(u, v)]
    max_support = max(nx.get_edge_attributes(data_graph, 'support').values())
    num_nodes = 7650#data_graph.number_of_nodes()
    #print(data_graph.number_of_nodes())
    similarity_matrix = np.zeros((num_nodes, num_nodes))  
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    labels = data['y']

    for u, v, data in data_graph.edges(data=True):
        u = int(u)
        v = int(v)
        # print(u, v)
        support = data['support']
        # print(support)
        similarity = (support + 1) / (max_support + 1)
        # print(similarity, type(u))
        similarity_matrix[u][v] = similarity
        similarity_matrix[v][u] = similarity

        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1
    similarity_matrix=torch.Tensor(similarity_matrix)
    adjacency_matrix=torch.Tensor(adjacency_matrix)
    num_classes = 8
    datasetname="photo"
    print(num_nodes)
    return similarity_matrix, adjacency_matrix, labels,data_graph,node_feature,num_classes,datasetname


def read_cs():
    print("cs")
    dataset=Coauthor(root='./tmp/cs',name='CS')
    data=dataset[0]
    node_feature=data['x']
    setattr(data, 'edges', data.edge_stores[0]['edge_index'].t())
    edges = [list([pair[0].item(),pair[1].item()]) for pair in data.edges]
    data_graph=nx.Graph()
    data_graph.add_edges_from(edges)

    edge_support = {}

    # 计算每条边出现在多少个三角形中
    for u, v in data_graph.edges():
        # 获取共同邻居（即形成三角形的第三个节点）
        common_neighbors = list(nx.common_neighbors(data_graph, u, v))
        support = len(common_neighbors)
        edge_support[(u, v)] = support
    for u, v in data_graph.edges():
        data_graph[u][v]['support'] = edge_support[(u, v)]
    max_support = max(nx.get_edge_attributes(data_graph, 'support').values())
    num_nodes = 18333#data_graph.number_of_nodes()
    print(data_graph.number_of_nodes())
    similarity_matrix = np.zeros((num_nodes, num_nodes))  
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    labels = data['y']

    for u, v, data in data_graph.edges(data=True):
        u = int(u)
        v = int(v)
        # print(u, v)
        support = data['support']
        # print(support)
        similarity = (support + 1) / (max_support + 1)
        # print(similarity, type(u))
        similarity_matrix[u][v] = similarity
        similarity_matrix[v][u] = similarity

        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1
    similarity_matrix=torch.Tensor(similarity_matrix)
    adjacency_matrix=torch.Tensor(adjacency_matrix)
    num_classes = 15
    datasetname = "cs"
    return similarity_matrix, adjacency_matrix, labels,data_graph,node_feature,num_classes,datasetname



def read_pubmed():
    dataset=Planetoid(root='./tmp/pubmed',name='PubMed')
    data=dataset[0]
    node_feature=data['x']
    setattr(data, 'edges', data.edge_stores[0]['edge_index'].t())
    edges = [list([pair[0].item(),pair[1].item()]) for pair in data.edges]
    data_graph=nx.Graph()
    data_graph.add_edges_from(edges)

    edge_support = {}

    # 计算每条边出现在多少个三角形中
    for u, v in data_graph.edges():
        # 获取共同邻居（即形成三角形的第三个节点）
        common_neighbors = list(nx.common_neighbors(data_graph, u, v))
        support = len(common_neighbors)
        edge_support[(u, v)] = support
    for u, v in data_graph.edges():
        data_graph[u][v]['support'] = edge_support[(u, v)]
    max_support = max(nx.get_edge_attributes(data_graph, 'support').values())
    num_nodes = 19717#data_graph.number_of_nodes()
    print(data_graph.number_of_nodes())
    similarity_matrix = np.zeros((num_nodes, num_nodes))  
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    labels = data['y']

    for u, v, data in data_graph.edges(data=True):
        u = int(u)
        v = int(v)
        # print(u, v)
        support = data['support']
        # print(support)
        similarity = (support + 1) / (max_support + 1)
        # print(similarity, type(u))
        similarity_matrix[u][v] = similarity
        similarity_matrix[v][u] = similarity

        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1
    similarity_matrix=torch.Tensor(similarity_matrix)
    adjacency_matrix=torch.Tensor(adjacency_matrix)
    num_classes = 3
    datasetname = "pubmed"
    return similarity_matrix, adjacency_matrix, labels,data_graph,node_feature,num_classes,datasetname


def read_citeseer():
    dataset=Planetoid(root='./tmp/citeseer',name='CiteSeer')
    data=dataset[0]
    node_feature=data['x']
    setattr(data, 'edges', data.edge_stores[0]['edge_index'].t())
    edges = [list([pair[0].item(),pair[1].item()]) for pair in data.edges]
    data_graph=nx.Graph()
    data_graph.add_edges_from(edges)

    edge_support = {}

    # 计算每条边出现在多少个三角形中
    for u, v in data_graph.edges():
        # 获取共同邻居（即形成三角形的第三个节点）
        common_neighbors = list(nx.common_neighbors(data_graph, u, v))
        support = len(common_neighbors)
        edge_support[(u, v)] = support
    for u, v in data_graph.edges():
        data_graph[u][v]['support'] = edge_support[(u, v)]
    max_support = max(nx.get_edge_attributes(data_graph, 'support').values())
    num_nodes = 3327#data_graph.number_of_nodes()
    print(data_graph.number_of_nodes())
    similarity_matrix = np.zeros((num_nodes, num_nodes))  
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    labels = data['y']

    for u, v, data in data_graph.edges(data=True):
        u = int(u)
        v = int(v)
        # print(u, v)
        support = data['support']
        # print(support)
        similarity = (support + 1) / (max_support + 1)
        # print(similarity, type(u))
        similarity_matrix[u][v] = similarity
        similarity_matrix[v][u] = similarity

        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1
    similarity_matrix=torch.Tensor(similarity_matrix)
    adjacency_matrix=torch.Tensor(adjacency_matrix)
    num_classes = 6
    datasetname = "citeseer"
    return similarity_matrix, adjacency_matrix, labels,data_graph,node_feature,num_classes,datasetname



def read_cora():
    dataset=Planetoid(root='./tmp/cora',name='Cora')
    data=dataset[0]
    node_feature=data['x']
    setattr(data, 'edges', data.edge_stores[0]['edge_index'].t())
    edges = [list([pair[0].item(),pair[1].item()]) for pair in data.edges]
    data_graph=nx.Graph()
    data_graph.add_edges_from(edges)

    edge_support = {}

    # 计算每条边出现在多少个三角形中
    for u, v in data_graph.edges():
        # 获取共同邻居（即形成三角形的第三个节点）
        common_neighbors = list(nx.common_neighbors(data_graph, u, v))
        support = len(common_neighbors)
        edge_support[(u, v)] = support
    for u, v in data_graph.edges():
        data_graph[u][v]['support'] = edge_support[(u, v)]
    max_support = max(nx.get_edge_attributes(data_graph, 'support').values())
    num_nodes = data_graph.number_of_nodes()

    similarity_matrix = np.zeros((num_nodes, num_nodes))  
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    labels = data['y']
    for u, v, data in data_graph.edges(data=True):
        u = int(u)
        v = int(v)
        # print(u, v)
        support = data['support']
        # print(support)
        similarity = (support + 1) / (max_support + 1)
        # print(similarity, type(u))
        similarity_matrix[u][v] = similarity
        similarity_matrix[v][u] = similarity

        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1
    similarity_matrix=torch.Tensor(similarity_matrix)
    adjacency_matrix=torch.Tensor(adjacency_matrix)
    num_classes = 7
    datasetname = "cora"
    return similarity_matrix, adjacency_matrix, labels,data_graph,node_feature,num_classes,datasetname

    






def read_email():
    data_graph = nx.Graph()
    labels = []
    with open('dataset/email-eu/email-Eu-core.txt', 'r') as file:
        for line in file:
            v = line.split()
            data_graph.add_edge(u_of_edge=int(v[0]), v_of_edge=int(v[1]))

    with open('dataset/email-eu/email-Eu-core-department-labels.txt', 'r') as file:
        for line in file:
            v = line.split()
            node_id = int(v[0])
            node_label = int(v[1])
            # print(v, int(v[0]), int(v[1]))
            data_graph.nodes[node_id]["labels"] = node_label  # 注意不能用label, 是固有标签
            labels.append(node_label)

    nx.write_gml(data_graph, 'dataset/email-eu/email-Eu-core.gml')
    print(labels)

    # print(data_graph)

    # 创建一个字典来存储边的支持度
    edge_support = {}

    # 计算每条边出现在多少个三角形中
    for u, v in data_graph.edges():
        # 获取共同邻居（即形成三角形的第三个节点）
        common_neighbors = list(nx.common_neighbors(data_graph, u, v))
        support = len(common_neighbors)
        edge_support[(u, v)] = support

    for u, v in data_graph.edges():
        data_graph[u][v]['support'] = edge_support[(u, v)]

    nx.write_gml(data_graph, 'dataset/email-eu/email-Eu-core-support.gml')


def get_similarity_matrix_email():
    data_graph = nx.read_gml(path='dataset/email-eu/email-Eu-core-support.gml')
    max_support = max(nx.get_edge_attributes(data_graph, 'support').values())
    print("max_support: ", max_support)
    num_nodes = data_graph.number_of_nodes()

    similarity_matrix = np.zeros((num_nodes, num_nodes))
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    labels = []

    for u, v, data in data_graph.edges(data=True):
        u = int(u)
        v = int(v)
        # print(u, v)
        support = data['support']
        # print(support)
        similarity = (support + 1) / (max_support + 1)
        # print(similarity, type(u))
        similarity_matrix[u][v] = similarity
        similarity_matrix[v][u] = similarity

        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1

    for node in data_graph.nodes(data=True):
        labels.append(node[1]["labels"])


    print("Similarity Matrix:")
    print(similarity_matrix)
    # print(similarity_matrix[0][0])

    print("Adjacency Matrix:")
    print(adjacency_matrix)

    print("Labels:")
    print(labels)
    similarity_matrix=torch.Tensor(similarity_matrix)
    adjacency_matrix=torch.Tensor(adjacency_matrix)
    node_feature = [[]]
    num_classes = 42
    labels=torch.tensor(labels)
    datasetname = "email-eu"
    return similarity_matrix, adjacency_matrix, labels,data_graph,node_feature,num_classes,datasetname

def get_high_level_graph(data_graph,adj):
    neg_graph = nx.Graph(data_graph)####
    neg_dict = {}
    elist3 = [(0, 1), (0, 2), (1, 2)]
    sg3 = gt.Graph(directed=False)
    sg3.add_edge_list(elist3)
    subgraph_dicts, orbit_partition_sizes = small_graph_prep([elist3])
    selected_edge_list = set()
    selected_node_list = set()
    flagdict = {}
    flagset = [[]]
    curnum = 0

    graph = gt.Graph(directed = False)
    graph.add_edge_list(data_graph.edges)

    aut_group = gt_topology.subgraph_isomorphism(sg3,graph, induced=False, subgraph=True, generator=False)
    selected_node_list, selected_edge_list, flagdict, curnum, flagset = selected_list_gen(aut_group,
                                                                                          selected_node_list,
                                                                                          selected_edge_list,
                                                                                          neg_graph,
                                                                                          neg_dict,
                                                                                          flagdict,
                                                                                          curnum,
                                                                                          flagset)
    high_pos_graph = nx.Graph(data_graph)
    # print(high_pos_graph)
    # print(selected_edge_list)
    # print( data_graph.edges)
    for edge in data_graph.edges:
        if (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))) not in selected_edge_list:
            high_pos_graph.remove_edge(edge[0], edge[1])
    nx.write_gml(data_graph,'cora_orig.gml')
    nx.write_gml(high_pos_graph,'cora_high.gml')
    target_matrix = torch.zeros((len(adj), len(adj)))
    for u, v, data in high_pos_graph.edges(data=True):
        u = int(u)
        v = int(v)

        target_matrix[u][v] = 1
        target_matrix[v][u] = 1
    target_matrix = target_matrix + torch.eye(len(adj))
    return target_matrix,high_pos_graph




def selected_list_gen(aut_group,selected_node_list,selected_edge_list,neg_graph,neg_dict,flagdict,curnum,flagset):#加一个参数做负样本

    curflag=0
    if(len(aut_group)==0):
        return selected_node_list,selected_edge_list
    k=len(aut_group[0]) #看是几plex结构
    for i in aut_group:
        temp=i.get_array()
        #处理负样本
        tup=tuple(np.sort(temp))
        if  tup not in neg_dict and (tup[0],tup[1]) in neg_graph.edges:#已经破坏的结构跳过
            neg_graph.remove_edge(tup[0],tup[1])
            neg_dict[tup]=1

        for j in range(k):
            if(temp[j] in flagdict.keys() and curflag==0):
                curflag=flagdict[temp[j]]
            selected_node_list.add(temp[j])
        if curflag==0:
            curnum+=1
            flagset.append([])
            curflag=curnum

        for j in range(k):
            if(temp[j] in flagdict.keys()):
                continue
            flagdict[temp[j]]=curflag
            flagset[curflag].append(temp[j])
        curflag=0

        for j in range(k-1):
            selected_edge_list.add((min(temp[j],temp[j+1]),max(temp[j],temp[j+1])))
        selected_edge_list.add((min(temp[0],temp[k-1]),max(temp[0],temp[k-1])))
    print("curnum",curnum)
    #print(flagdict)
    #print(flagset)
    return selected_node_list,selected_edge_list,flagdict,curnum,flagset
def small_graph_prep(edge_lists):
    subgraph_dicts = []
    orbit_partition_sizes = []
    for edge_list in edge_lists:
        subgraph, orbit_partition, orbit_membership, aut_count = \
            induced_edge_automorphism_orbits(edge_list=edge_list,
                            directed=False,
                            directed_orbits=False)
        subgraph_dicts.append({'subgraph': subgraph, 'orbit_partition': orbit_partition,
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))
    return subgraph_dicts,orbit_partition_sizes


def induced_edge_automorphism_orbits(edge_list, **kwargs):
    ##### induced edge automorphism orbits (according to the vertex automorphism group) #####

    directed = kwargs['directed'] if 'directed' in kwargs else False #undirected
    directed_orbits = kwargs['directed_orbits'] if 'directed_orbits' in kwargs else False

    graph, orbit_partition, orbit_membership, aut_count = automorphism_orbits(edge_list=edge_list,
                                                                              directed=directed,
                                                                              print_msgs=False)
    edge_orbit_partition = dict()
    edge_orbit_membership = dict()
    edge_orbits2inds = dict()
    ind = 0

    if not directed:
        edge_list = to_undirected(torch.tensor(graph.get_edges()).transpose(1, 0)).transpose(1, 0).tolist()

    # infer edge automorphisms from the vertex automorphisms
    for i, edge in enumerate(edge_list):
        if directed_orbits:
            edge_orbit = (orbit_membership[edge[0]], orbit_membership[edge[1]])
        else:
            edge_orbit = frozenset([orbit_membership[edge[0]], orbit_membership[edge[1]]])
        if edge_orbit not in edge_orbits2inds:
            edge_orbits2inds[edge_orbit] = ind
            ind_edge_orbit = ind
            ind += 1
        else:
            ind_edge_orbit = edge_orbits2inds[edge_orbit]

        if ind_edge_orbit not in edge_orbit_partition:
            edge_orbit_partition[ind_edge_orbit] = [tuple(edge)]
        else:
            edge_orbit_partition[ind_edge_orbit] += [tuple(edge)]

        edge_orbit_membership[i] = ind_edge_orbit

    return graph, edge_orbit_partition, edge_orbit_membership, aut_count



def automorphism_orbits(edge_list, print_msgs=True, **kwargs):
    ##### vertex automorphism orbits #####

    directed = kwargs['directed'] if 'directed' in kwargs else False

    graph = gt.Graph(directed=directed)
    graph.add_edge_list(edge_list)


    # compute the vertex automorphism group
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v

    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role

    orbit_membership_list = [[], []]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse=True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i, vertex in enumerate(orbit_membership_list[0])}

    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit] + [vertex]

    aut_count = len(aut_group)

    return graph, orbit_partition, orbit_membership, aut_count
