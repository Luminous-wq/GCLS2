import n2
import argparses
import torch
import numpy as np
from read_datasets import get_similarity_matrix_email, get_high_level_graph,read_cora,read_citeseer,read_pubmed,read_cs,read_photo
from model import GcnNet, DNN,MLP
from torch_geometric.graphgym import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid,KarateClub,Amazon
from torch_geometric.utils import dropout_adj
from sklearn import metrics
import time
from sklearn.metrics import f1_score

####nNCLANCLANCLANCLANCLANCALNCAL
def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    l1 = nei_con_loss(z1, z2, tau, adj, hidden_norm)
    l2 = nei_con_loss(z2, z1, tau, adj, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret
# def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
#     if hidden_norm:
#         z1 = F.normalize(z1)
#         z2 = F.normalize(z2)
#     return torch.mm(z1, z2.t())


def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
            intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    loss = loss / nei_count  # divided by the number of positive pairs for each node


    return -torch.log(loss)
def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

#_---------------------------------------------------------------------------------
#GRACE GRACE GRACE

# def sim(z1, z2):
#     # normalize embeddings across feature dimension
#     z1 = F.normalize(z1)
#     z2 = F.normalize(z2)

#     s = torch.mm(z1, z2.t())
#     return s
def get_loss(z1, z2):
    # calculate SimCLR loss
    f = lambda x: torch.exp(x / 1.0)

    refl_sim = f(sim(z1, z1))  # intra-view pairs
    between_sim = f(sim(z1, z2))  # inter-view pairs

    # between_sim.diag(): positive pairs
    x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
    loss = -torch.log(between_sim.diag() / x1)

    return loss

def final_loss(z1,z2):
    l1 = get_loss(z1, z2)
    l2 = get_loss(z2, z1)

    ret = (l1 + l2) * 0.5

    return ret.mean()

#--------------------------------------------------------------------------
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

if __name__ == '__main__':
    args = argparses.args_parser()
    args.cuda = args.cuda and torch.cuda.is_available()
    set_seed(args)

    #TODO datalodar
    X,A,labels,data_graph,node_feature,num_classes,dataset = read_cora()
    print("ok, loading")
    _, _tt, idx_train, idx_test_val = train_test_split(labels, list(range(len(labels))), test_size=0.2, random_state=42)
    _, _, idx_test, idx_val = train_test_split(_tt, idx_test_val, test_size=0.5, random_state=42)
    id_train = torch.LongTensor(idx_train)
    id_val = torch.LongTensor(idx_val)
    id_test = torch.LongTensor(idx_test)




    cl_epoch_num = 50
    epoch_num = 500
    best_loss = 1e9
    A2 = drop_feature(A,0.3)
    X2 = drop_feature(X,0.3)
    target_matrix,target_graph = get_high_level_graph(data_graph,A)
    loss_fn = torch.nn.CrossEntropyLoss()
    avg_acc = 0
    avg_nmi = 0
    avg_f1 = 0
    max_acc = 0
    max_nmi = 0
    max_f1 = 0
    min_acc = 1 
    min_nmi = 1
    min_f1 = 1
    X=node_feature
   
    for i in range(10):
        GCN1 = GcnNet(input_dim=len(X[0]), hidden_dim=64, output_dim=num_classes)
        GCN2 = GcnNet(input_dim=len(X[0]), hidden_dim=64, output_dim=num_classes)
        dnn = DNN(len(X[0]),64,32)
        dnn2 = DNN(len(node_feature[0]),64,32)
                        #cora 0.0005
        optimizer = optim.Adam(list(GCN1.parameters())+list(dnn.parameters())+list(dnn2.parameters()), lr=0.0005)
        optimizer2 = optim.Adam(list(GCN1.parameters()), lr=0.01)
        best_loss_cl = 1e9
        max_acc_cl = 0
        start = time.time()
        for epoch in range(cl_epoch_num):
            optimizer.zero_grad()
            GCN1.train()
            #temp=torch.cat((dnn(X),dnn2(node_feature)),1)
            temp=X
            feature = GCN1(A,temp)
            feature2 = GCN1(target_matrix,temp)
            # temp_feature = torch.softmax(feature, dim=1)
            # print(feature[id_train].size(), torch.tensor(labels)[id_train].size())
            #loss = loss_fn(feature[id_train], torch.tensor(labels)[id_train])
            #loss = 0.2*final_loss(feature,feature2)#,target_matrix)
            loss = contrastive_loss(feature,feature2,target_matrix,tau=1)
            #loss = contrastive_loss(feature,feature2,A)
            loss.backward()
            optimizer.step()
            print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, cl_epoch_num, loss.item()))
            if (epoch + 1) % 10 == 0:
                GCN1.eval()
                feature = GCN1(A,temp)
                pred = torch.argmax(feature, dim=1)
                correct = torch.sum(pred[id_test] == torch.tensor(labels)[id_test])
                acc = correct.item() / len(id_test)
                
                print()
                print('Accuracy on test set: {:.4f}'.format(acc))
                print()

                valid_loss_cl = loss_fn(feature[id_val], labels[id_val])
                if valid_loss_cl < best_loss_cl:
                    best_loss_cl = valid_loss_cl
                    best_epoch = epoch

                    torch.save(GCN1.state_dict(), 'best_model_cl_' + dataset + ".pt")
                    print(best_loss_cl)
                    print("Model saved")

        m_acc = 0
        m_nmi = 0
        m_f1 = 0
        best_loss = 1e9
        GCN1.load_state_dict(torch.load('best_model_cl_' + dataset + ".pt"))
        for epoch in range(epoch_num):
            optimizer.zero_grad()
            GCN1.train()
            #temp=torch.cat((mlp(X),mlp2(node_feature)),1)
            #temp=torch.cat((dnn(X),dnn2(node_feature)),1)
            temp=X
            feature = GCN1(A,temp)
            #feature2 = GCN1(target_matrix,X)

            # temp_feature = torch.softmax(feature, dim=1)
            # print(feature[id_train].size(), torch.tensor(labels)[id_train].size())
            loss = loss_fn(feature[id_train], labels[id_train])
            #loss += 0.2*final_loss(feature,feature2)
            loss.backward()
            optimizer.step()
            print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, epoch_num, loss.item()))
            if (epoch + 1) % 10 == 0:

                GCN1.eval()
                feature = GCN1(A,temp)
                pred = torch.argmax(feature, dim=1)
                correct = torch.sum(pred[id_test] == torch.tensor(labels)[id_test])
                acc = correct.item() / len(id_test)
                nmi = metrics.normalized_mutual_info_score(pred[id_test].cpu().numpy(),torch.tensor(labels)[id_test].cpu().numpy())
                f1 = f1_score(pred[id_test].cpu().numpy(),torch.tensor(labels)[id_test].cpu().numpy(), average='weighted')
                m_acc = max(acc,m_acc)
                m_nmi = max(nmi,m_nmi)
                m_f1 = max(f1,m_f1)
                print()
                print('Accuracy on test set: {:.4f}'.format(acc))
                print('NMI on test set: {:.4f}'.format(nmi))
                print('F1 on test set: {:.4f}'.format(f1))
                print()

                valid_loss = loss_fn(feature[id_val], labels[id_val])
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch

                    torch.save(GCN1.state_dict(), 'best_model_' + dataset + ".pt")
                    print("Model saved")
        end = time.time()
        print('time')
        print(end-start)
        GCN1.load_state_dict(torch.load('best_model_' + dataset + ".pt"))
        feature = GCN1(A,X)
        pred = torch.argmax(feature, dim=1)
        correct = torch.sum(pred[id_test] == torch.tensor(labels)[id_test])
        m_acc = correct.item() / len(id_test)
        m_nmi = metrics.normalized_mutual_info_score(pred[id_test].cpu().numpy(),torch.tensor(labels)[id_test].cpu().numpy())
        m_f1 = f1_score(pred[id_test].cpu().numpy(),torch.tensor(labels)[id_test].cpu().numpy(), average='weighted')


        avg_acc += m_acc
        avg_nmi += m_nmi
        avg_f1 += m_f1
        max_acc = max(max_acc,m_acc)
        max_nmi = max(max_nmi,m_nmi)
        max_f1 = max(max_f1,m_f1)
        min_nmi = min(min_nmi,m_nmi)
        min_acc = min(min_acc,m_acc)
        min_f1 = min(min_f1,m_f1)
        print('Final Accuracy on test set: {:.4f}'.format(m_acc))
        print('Final NMI on test set: {:.4f}'.format(m_nmi))
        print('Final F1 on test set: {:.4f}'.format(m_f1))
    print('avg')
    print(avg_acc/10)
    print(avg_nmi/10)
    print(avg_f1/10)
    print("max")
    print(max_acc)
    print(max_nmi)
    print(max_f1)
    print("min")
    print(min_acc)
    print(min_nmi)
    print(min_f1)








    


    


