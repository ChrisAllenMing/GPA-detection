import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nproposal, nfeat, dropout = 0.5):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.conv_1 = GraphConvolution(nfeat, nfeat)
        self.emb = GraphConvolution(nfeat, nfeat)
        # self.weight = GraphConvolution(nfeat, 1)
        self.weight = GraphConvolution(1, 1)

    def forward(self, x, w, adj):
        feat_1 = F.relu(self.conv_1(x, adj))
        feat_1 = F.dropout(feat_1, self.dropout, training=self.training)
        emb = self.emb(feat_1, adj)
        # weight = F.sigmoid(self.weight(feat_1, adj))
        # weight = torch.mm(adj, w)
        weight = w

        output = emb * weight
        # output = torch.sum(output, dim = 0) / torch.sum(weight)

        return output, weight


class GCN_pooling(nn.Module):
    def __init__(self, nproposal, nfeat, dropout = 0.5):
        super(GCN, self).__init__()
        self.dropout = dropout

        # the first group
        nproposal_1 = nproposal / 4
        self.emb_1 = GraphConvolution(nfeat, nfeat)
        self.pool_1 = GraphConvolution(nfeat, nproposal_1)

        # the second group
        nproposal_2 = nproposal / 16
        self.emb_2 = GraphConvolution(nfeat, nfeat)
        self.pool_2 = GraphConvolution(nfeat, nproposal_2)

        # the third group
        nproposal_3 = 1
        self.emb_3 = GraphConvolution(nfeat, nfeat)
        self.pool_3 = GraphConvolution(nfeat, nproposal_3)

    def forward(self, x, adj):
        # the first group
        input_1 = F.relu(self.emb_1(x, adj))
        #input_1 = F.dropout(input_1, self.dropout, training = self.training)
        assign_1 = F.softmax(F.relu(self.pool_1(x, adj)), dim = 1)
        #assign_1 = F.dropout(assign_1, self.dropout, training = self.training)
        output_1 = torch.mm(assign_1.t(), input_1)
        adj_1 = torch.mm(torch.mm(assign_1.t(), adj), assign_1)

        # the second group
        input_2 = F.relu(self.emb_2(output_1, adj_1))
        #input_2 = F.dropout(input_2, self.dropout, training = self.training)
        assign_2 =F.softmax(F.relu(self.pool_2(output_1, adj_1)), dim = 1)
        #assign_2 = F.dropout(assign_2, self.dropout, training = self.training)
        output_2 = torch.mm(assign_2.t(), input_2)
        adj_2 = torch.mm(torch.mm(assign_2.t(), adj_1), assign_2)

        # the third group 
        input_3 = self.emb_3(output_2, adj_2)
        assign_3 = F.softmax(self.pool_3(output_2, adj_2), dim = 1)
        output_3 = torch.mm(assign_3.t(), input_3)

        # get the link and entropy regularization loss
        gcn_loss_link = 0
        gcn_loss_entropy = 0

        gcn_loss_link = gcn_loss_link + torch.pow(adj - torch.mm(assign_1, assign_1.t()), 2).mean()
        gcn_loss_link = gcn_loss_link + torch.pow(adj_1 - torch.mm(assign_2, assign_2.t()), 2).mean()
        gcn_loss_link = gcn_loss_link + torch.pow(adj_2 - torch.mm(assign_3, assign_3.t()), 2).mean()

        entropy_1 = -torch.mul(assign_1, torch.log(assign_1))
        gcn_loss_entropy = gcn_loss_entropy + torch.mean(torch.sum(entropy_1, dim=1), dim=0)
        entropy_2 = -torch.mul(assign_2, torch.log(assign_2))
        gcn_loss_entropy = gcn_loss_entropy + torch.mean(torch.sum(entropy_2, dim=1), dim=0)
        entropy_3 = -torch.mul(assign_3, torch.log(assign_3))
        gcn_loss_entropy = gcn_loss_entropy + torch.mean(torch.sum(entropy_3, dim=1), dim=0)

        return output_3.view(-1), gcn_loss_link, gcn_loss_entropy
