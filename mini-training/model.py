import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import conv

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight) 
        
        if self.use_bias:
            init.zeros_(self.bias)
    
    def forward(self, adjacency, input_feature):
        # print(input_feature, self.weight)
        support = torch.mm(input_feature, self.weight)
        adjacency = adjacency.type(torch.float32)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output
 
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GcnNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.gcn1 = GraphConvolution(32, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)
    
    def forward(self, adjacency, feature):
        feature = self.layer1(feature)
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, X):
        X = self.layer1(X)
        return X


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)


class GSCLedge(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp):
        super(GSCLedge, self).__init__()
        #self.encoder = GcnNet(in_dim, hid_dim, act_fn)
        self.encoder = conv.GCNConv(in_dim,act_fn)
        self.temp = temp
        #self.proj = MLP(hid_dim, out_dim)
        self.proj = MLP(act_fn, out_dim)
        

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = torch.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: torch.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -torch.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, edge_index, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(feat, edge_index)

        return h#.detach()

    def forward(self, edge1, edge2, feat1, feat2):
        # encoding
        h1 = self.encoder(feat1, edge1)
        h2 = self.encoder(feat2, edge2)

        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)

        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        return ret.mean()

class GSCL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp):
        super(GSCL, self).__init__()
        self.encoder = GcnNet(in_dim, hid_dim, act_fn)
        self.temp = temp
        #self.proj = MLP(hid_dim, out_dim)
        self.proj = MLP(act_fn, out_dim)
        

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = torch.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: torch.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -torch.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, feat)

        return h#.detach()

    def forward(self, adj1, adj2, feat1, feat2):
        # encoding
        h1 = self.encoder(adj1, feat1)
        h2 = self.encoder(adj2, feat2)

        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)

        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        return ret.mean()

class GSCL_motiv(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, act_fn, temp):
        super(GSCL_motiv, self).__init__()
        self.encoder = GcnNet(in_dim, hid_dim, act_fn)
        # self.encoder = DNN(in_dim, hid_dim, act_fn)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        # print(z1, z2.t())
        s = torch.mm(z1, z2.t())
        return s

    def get_loss(self, z1):
        # calculate SimCLR loss
        f = lambda x: torch.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs

        x1 = refl_sim.sum(1) + refl_sim.diag()
        # x1 = refl_sim.sum(1)
        loss = -torch.log(refl_sim.diag() / x1)

        return loss

    def get_embedding(self, adj1, feat1):
        # get embeddings from the model for evaluation
        h = self.encoder(adj1, feat1)
        # h = self.encoder(feat1)

        return h.detach()

    def forward(self, adj1, feat1):
        # encoding
        h1 = self.encoder(adj1, feat1)
        # h1 = self.encoder(feat1)
        # projection
        z1 = self.proj(h1)

        # get loss
        l1 = self.get_loss(z1)

        return l1.mean()