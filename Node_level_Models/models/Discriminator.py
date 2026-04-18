#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix


class FakeGraphDiscriminator(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False):

        super(FakeGraphDiscriminator, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, 1)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay
        
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # Loss function
        self.criterion = nn.BCELoss()

    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x) 
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return torch.sigmoid(x)

    def train_step(self, local_data, device, fake_graph, train_iters=2):
        for i in range(train_iters):
            self.train()
            self.optimizer.zero_grad()
            x_norm = F.normalize(
                local_data.x.clone().to(device),
                p=2, 
                dim=1
            )
            pred_real = self(x_norm, local_data.edge_index.to(device), local_data.edge_weight.to(device))
            real_labels = torch.ones(pred_real.size(0), 1, device=device)

            pred_fake = self(fake_graph.x.to(device), fake_graph.edge_index.to(device), fake_graph.edge_weight.to(device))                                                                                                              
            fake_labels = torch.zeros(pred_fake.size(0), 1, device=device)

            loss_real = self.criterion(pred_real, real_labels)
            loss_fake = self.criterion(pred_fake, fake_labels)
            loss = loss_real + loss_fake

            loss.backward()
            self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    num_nodes = 5
    num_features = 10
    x = torch.randn((num_nodes, num_features))

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FakeGraphDiscriminator(
        nfeat=num_features,
        nhid=16,
        dropout=0.5,
        layer=2,
        device=device,
        layer_norm_first=True,
        use_ln=True
    ).to(device)

    x = x.to(device)
    edge_index = edge_index.to(device)
    output = model(x, edge_index)

    print("Shape of input node features:", x.shape)
    print("Shape of edge index:", edge_index.shape)
    print("Shape of output:", output.shape)
    print("Output values (probability of each node being a pseudo-node):")
    print(output)