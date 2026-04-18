import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import os
import yaml

class MultiFedKD_Generator(nn.Module):
    def __init__(self, noise_dim, feat_dim, out_dim, dropout, args, combine_mode='concat', k_generators=3):
        super(MultiFedKD_Generator, self).__init__()
        self.noise_dim = noise_dim
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.combine_mode = combine_mode
        self.k_generators = k_generators
        
        # Calculate the number of classes each generator is responsible for
        self.classes_per_generator = out_dim // k_generators
        print("out_dim: ", out_dim, "k_generators: ", k_generators, "self.classes_per_generator: ", self.classes_per_generator)
        if out_dim % k_generators != 0:
            print(f"Warning: {out_dim} classes cannot be evenly divided by {k_generators} generators")
        
        self.generators = nn.ModuleList()
        for i in range(k_generators):
            # The output dimension of each generator is the number of classes it is responsible for
            gen_out_dim = self.classes_per_generator
            # If it's the last generator, remaining classes may need to be handled
            if i == k_generators - 1 and out_dim % k_generators != 0:
                gen_out_dim = out_dim - i * self.classes_per_generator
            
            generator = SingleGenerator(
                noise_dim=noise_dim,
                feat_dim=feat_dim,
                out_dim=gen_out_dim,
                dropout=dropout,
                args=args,
                combine_mode=combine_mode
            )
            self.generators.append(generator)
    
    def forward(self, z, c):
        """
        Forward propagation
        z: Noise vector [B, noise_dim]
        c: Conditional label [B,]
        """
        B = z.size(0)
        
        # Initialize output
        node_feats = torch.zeros(B, self.feat_dim, device=z.device)
        adj_matrix = torch.zeros(B, B, device=z.device)
        
        # Used to store the output of each generator (for debugging or subsequent processing)
        all_node_feats = []
        all_adj_matrices = []
        
        # Process the corresponding classes for each generator
        for i, generator in enumerate(self.generators):
            # Calculate the class range assigned to the current generator
            start_idx = i * self.classes_per_generator
            end_idx = start_idx + generator.out_dim
            
            # Create a mask to select samples belonging to the classes assigned to the current generator
            mask = (c >= start_idx) & (c < end_idx)
            
            if mask.sum() > 0:
                c_relative = c[mask] - start_idx
                
                z_masked = z[mask]
                node_feats_masked, adj_matrix_masked, _ = generator(z_masked, c_relative)
                
                node_feats[mask] = node_feats_masked
                
                adj_matrix[mask, :] = 0 
                adj_matrix[:, mask] = 0
                adj_matrix[mask.unsqueeze(1) & mask.unsqueeze(0)] = adj_matrix_masked.flatten()
                
                all_node_feats.append(node_feats_masked)
                all_adj_matrices.append(adj_matrix_masked)
        
        return node_feats, adj_matrix, None


class SingleGenerator(nn.Module):
    """Single generator, with the same structure as the original FedKD_Generator"""
    def __init__(self, noise_dim, feat_dim, out_dim, dropout, args, combine_mode='concat'):
        super(SingleGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.combine_mode = combine_mode

        self.emb_layer = nn.Embedding(out_dim, out_dim)

        # Determine the input dimension based on the concatenation method
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
        print(f"Generator dims for {out_dim} classes: {dims}")

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

        # Node feature generation uses mean and variance, supporting reparameterization
        self.nodes_mu = nn.Linear(dims[-1], self.feat_dim)
        self.nodes_logvar = nn.Linear(dims[-1], self.feat_dim)

        # Edge predictor
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self._init_weights()
    
    def reparameterize(self, mu, logvar, noise_scale=1):
        """Adjust noise intensity via noise_scale to enhance diversity"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_scale
        return mu + eps * std
    
    def predict_edges(self, node_features):
        """Calculate edge probabilities for all node pairs via dot product"""
        encoded_z = self.edge_encoder(node_features)
        dot_product = torch.matmul(encoded_z, encoded_z.T)
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

        adj_logits = self.predict_edges(node_feats)
        T = 1
        adj_matrix = torch.sigmoid(adj_logits * T)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2 

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
    noise_dim = 16
    feat_dim = 32
    out_dim = 10
    dropout = 0.1
    
    class Args:
        def __init__(self):
            self.dataset = 'Cora'
    
    args = Args()

    # Create multi-generator instances (3 generators)
    k_generators = 3
    model = MultiFedKD_Generator(noise_dim, feat_dim, out_dim, dropout, args, 
                               combine_mode='concat', k_generators=k_generators)


    # Simulate input: Generate 12 pseudo nodes
    B = 22
    z = torch.randn(B, noise_dim)
    c = torch.randint(0, out_dim, (B,))

    print(f"Labels: {c.tolist()}")
    print(f"Generator 0 is responsible for classes: 0-{model.classes_per_generator - 1}")
    print(f"Generator 1 is responsible for classes: {model.classes_per_generator}-{2 * model.classes_per_generator - 1}")
    print(f"Generator 2 is responsible for classes: {2 * model.classes_per_generator}-{out_dim - 1}")
    
    # ====== Invoke the model ======
    node_feats, adj_matrix, _ = model(z, c)
    print("adj_matrix.shape: ", adj_matrix.shape)
    print("adj_matrix: ", adj_matrix)
    print("node_feats.shape: ", node_feats.shape)
    print("node_feats: ", node_feats)