import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class Classifier(nn.Module):
    def __init__(self, nhid, nclass, lr=0.01, weight_decay=5e-4, device=None):
        super(Classifier, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.gc2 = GCNConv(nhid, nclass)
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, x, edge_index, edge_weight=None):
        output = self.gc2(x, edge_index, edge_weight)
        return F.log_softmax(output,dim=1)
    
    def train_with_proto(self, local_proto_list, train_iters=5):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        embeddings = []
        labels = []
        edge_index = torch.tensor([[], []], dtype=torch.long)  # 0 -> 1 和 1 -> 0
        edge_weight = torch.tensor([], dtype=torch.float)
        for j in range(len(local_proto_list)):
            local_proto = local_proto_list[j]
            for label, proto in local_proto.items():
                embeddings.append(proto)
                labels.append(label)
        emb_stacked = torch.stack(embeddings, dim=0).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(emb_stacked, edge_index, edge_weight)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
        return loss.item()

    def _train_without_val(self, global_model, labels, idx_train, train_iters, verbose, agg_global_proto, args):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        batch_count = 0

        for i in range(train_iters):
            optimizer.zero_grad()
            output, proto = self.forward(self.features, self.edge_index, self.edge_weight)
            local_proto_label = {}
            loss_train = F.nll_loss(output[idx_train], labels[idx_train].squeeze())
            loss = loss_train        
            loss.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
            batch_count += 1
            
        return loss.item(), local_proto_label


