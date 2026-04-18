import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

class FedKD_Generator_single_class(nn.Module):
    def __init__(self, noise_dim, feat_dim, out_dim, sample_num, dropout):
        super(FedKD_Generator_single_class, self).__init__()
        self.noise_dim = noise_dim
        self.feat_dim = feat_dim
        self.sample_num = sample_num

        self.emb_layer = nn.Embedding(out_dim, out_dim)
        dims = [noise_dim + out_dim, 64, 128, 256]

        self.hid_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i+1]
            self.hid_layers.append(nn.Linear(d_in, d_out))
            self.hid_layers.append(nn.Tanh())
            self.hid_layers.append(nn.Dropout(dropout))
        self.nodes_layer = nn.Linear(256, self.feat_dim * self.sample_num)

        self.edge_layer = nn.ModuleList()
        self.edge_layer.append(nn.Linear(256, self.sample_num * self.sample_num))
        self.edge_layer.append(nn.Sigmoid())



    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer(c), z), dim=-1)
        hid = z_c
        for layer in self.hid_layers:
            hid = layer(hid)

        node_feats = self.nodes_layer(hid)  # [B, feat_dim]
        node_feats = node_feats.view(-1, self.sample_num, self.feat_dim)    #[B, sample_num, feat_dim]
        node_feats = F.normalize(node_feats, p=2, dim=2)

        node_edges = hid
        for layer in self.edge_layer:
            node_edges = layer(node_edges)
        node_edges = node_edges.view(-1, self.sample_num, self.sample_num)  # [B, sample_num, sample_num]
        node_edges = (node_edges + node_edges.transpose(1, 2)) / 2

        return node_feats, node_edges



if __name__ == '__main__':
    noise_dim = 16
    feat_dim = 1024
    out_dim = 5
    dropout = 0.1
    sample_dim = 10
    batch_size = 2

    model = FedKD_Generator_single_class(noise_dim, feat_dim, out_dim, sample_dim, dropout)

    z = torch.randn(batch_size, noise_dim)
    label_list = []
    for i in range(batch_size):
        label_list.append(i)

    c = torch.tensor(label_list)

    node_feats, adj_matrix = model(z, c)
    print("adj_matrix: \n")
    print(adj_matrix)
    threshold = 0.6
    adj_binary = (adj_matrix > threshold).float()
    adj_binary = torch.triu(adj_binary, diagonal=1)

    print("Node feature shape:", node_feats.shape)      # [10, 32] 
    print("Sample node features:", node_feats)
    print("Adjacency matrix shape:", adj_binary.shape)      # [10, 10]
    print("Sample adjacency matrix:\n", adj_binary)

    # ====== Visualize the adjacency matrix ======
    plt.figure(figsize=(6, 5))
    sns.heatmap(adj_binary.squeeze(0).detach().numpy(), cmap='Blues', annot=True, fmt=".2f")
    plt.title("Generated Pseudo Adjacency Matrix")
    plt.xlabel("Node j")
    plt.ylabel("Node i")
    plt.show()
