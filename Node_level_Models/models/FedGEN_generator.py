import torch
import torch.nn as nn

class FedGEN_ConGenerator(nn.Module):

    def __init__(self, noise_dim, hidden_dim, class_dim, dropout):
        super(FedGEN_ConGenerator, self).__init__()
        
        self.emb_layer = nn.Embedding(class_dim, noise_dim)
        self.hid_layers = nn.ModuleList()
        dims = [noise_dim * 2, hidden_dim]
        
        for i in range(len(dims) - 1):
            d_in = dims[i]
            d_out = dims[i + 1]
            self.hid_layers.append(nn.Linear(d_in, d_out))
            self.hid_layers.append(nn.Tanh())
            self.hid_layers.append(nn.Dropout(p=dropout, inplace=False))
        
        self.nodes_layer = nn.Linear(dims[-1], hidden_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer(c), z), dim=-1)
        for layer in self.hid_layers:
            z_c = layer(z_c)
        node_logits = self.nodes_layer(z_c)
        return node_logits
