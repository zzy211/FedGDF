import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import os
import yaml

class FedKD_Generator(nn.Module):
    def __init__(self, noise_dim, feat_dim, out_dim, dropout, args, combine_mode='concat'):
        super(FedKD_Generator, self).__init__()
        self.noise_dim = noise_dim
        self.feat_dim = feat_dim
        self.combine_mode = combine_mode

        self.emb_layer = nn.Embedding(out_dim, out_dim)

        if combine_mode == 'concat':
            input_dim = noise_dim + out_dim
        elif combine_mode == 'dot_product':
            input_dim = noise_dim
            assert noise_dim == out_dim, "for dot_product, noise_dim must equal out_dim, please reassign args.noise_dim to n_class."
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")
        
        config_file = "yaml/model_dims.yaml"
        if not os.path.exists(config_file):
            raise ValueError(f"The configuration file does not exist: {config_file}")
        with open(config_file, 'r') as f:
            dataset_configs = yaml.safe_load(f)['dataset_configs']
        dims = dataset_configs[args.dataset]['generator_dims']  
        dims = [input_dim] + dims 
        print("dims: ", dims)
        
        # dims = [input_dim, 64, 128, 256, 512]
        hid_layers = []
        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i+1]
            hid_layers += [
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.Tanh(),
                nn.Dropout(dropout)
            ]
        self.hid_layers = nn.Sequential(*hid_layers)

        # Node features are generated using mean and variance, supporting reparameterization
        self.nodes_mu = nn.Linear(dims[-1], self.feat_dim)
        self.nodes_logvar = nn.Linear(dims[-1], self.feat_dim)

        # New: Edge Predictor
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self._init_weights()
    
        
    def reparameterize(self, mu, logvar, noise_scale=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_scale
        return mu + eps * std
    
    def predict_edges(self, node_features):
        encoded_z = self.edge_encoder(node_features)  # [B, 32]
        dot_product = torch.matmul(encoded_z, encoded_z.T)  # [B, B]
        return dot_product

    def forward(self, z, c):
        B = z.size(0)
        c_emb = self.emb_layer(c)

        if self.combine_mode == "concat":
            z_c = torch.cat((c_emb, z), dim=-1)
        elif self.combine_mode == "dot_product":
            z_c = c_emb * z
        else:
            raise ValueError(f"Unknown combine mode {self.combine_mode}")

        hid = self.hid_layers(z_c)

        mu = self.nodes_mu(hid)
        logvar = self.nodes_logvar(hid)
        node_feats = self.reparameterize(mu, logvar)
        node_feats = F.normalize(node_feats, p=2, dim=1)

        # Compute edge probability matrix via dot product
        adj_logits = self.predict_edges(node_feats)
        T = 1
        adj_matrix = torch.sigmoid(adj_logits * T)  # Convert to probabilities (between 0 and 1)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Ensure symmetry

        return node_feats, adj_matrix, z_c
    
    def _init_weights(self):
        # Initialize hidden layer weights
        for layer in self.hid_layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='tanh')
                init.normal_(layer.bias, mean=0, std=0.01)
        
        # Initialize node layer weights
        init.kaiming_normal_(self.nodes_mu.weight, mode='fan_in', nonlinearity='linear')
        init.normal_(self.nodes_mu.bias, mean=0, std=0.01)
        init.kaiming_normal_(self.nodes_logvar.weight, mode='fan_in', nonlinearity='linear')
        init.normal_(self.nodes_logvar.bias, mean=0, std=0.01)
        
        # Initialize edge predictor weights
        init.kaiming_normal_(self.edge_encoder[0].weight, mode='fan_in', nonlinearity='relu')
        init.normal_(self.edge_encoder[0].bias, mean=0, std=0.01)
        init.xavier_normal_(self.edge_encoder[2].weight, gain=nn.init.calculate_gain('tanh'))
        init.normal_(self.edge_encoder[2].bias, mean=0, std=0.01)
        
        # Initialize embedding layer weights
        init.normal_(self.emb_layer.weight, mean=0, std=0.01)
    
    

if __name__ == '__main__':
    # ====== Model Parameters & Data Settings ======
    noise_dim = 16
    feat_dim = 32
    out_dim = 5 
    dropout = 0.1

    model = FedKD_Generator(noise_dim, feat_dim, out_dim, dropout)

    B = 10
    z = torch.randn(B, noise_dim)
    c = torch.randint(0, out_dim, (B,))

    node_feats, adj_matrix, z_c = model(z, c)
    print("adj_matrix.shape: ", adj_matrix.shape)
    print("adj_matrix: \n")
    print(adj_matrix)
    print("node_feats.shape: ", node_feats.shape)
    print("node_feats: \n")
    print(node_feats)