#This is zzy's code, good luck!
import torch
import random
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Reddit,Yelp
from torch_geometric.datasets import Coauthor, Amazon
# import Node_level_Models.helpers.selection_utils  as hs
from Node_level_Models.helpers.func_utils import  get_split, get_total_size, agg_local_proto_func, agg_global_proto_func, visualize_node_embeddings, visualize_node_embeddings_new_data, ContrastiveDiversityLoss, avg_per_class_acc
from Node_level_Models.helpers.cache_utils import PseudoGraphCache
from torch_geometric.utils import to_undirected
from Node_level_Models.helpers.split_graph_utils import split_Random, split_Louvain, split_Metis, split_dirichlet,split_graph_kernal
from Node_level_Models.models.construct import model_construct
from Node_level_Models.data.datasets import  ogba_data,Amazon_data,Coauthor_data
from Node_level_Models.aggregators.aggregation import fed_avg, fed_cls
from collections import deque
import os
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch.optim import Adam
from torch_geometric.data import Data
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from collections import OrderedDict
import yaml
from torch_scatter import scatter_add
from Node_level_Models.helpers.gens import *
from sklearn.mixture import GaussianMixture
from torch_geometric.utils import subgraph
from torch_geometric.datasets import CoraFull
import time

class MoveAvg:
    def __init__(self, size=10):
        self.size = size
        self.list_queue = deque()
        self.sum = 0
    
    def add_num(self, number):
        if(len(self.list_queue) >= self.size):
            left_number = self.list_queue.popleft()
            self.sum -= left_number
        self.list_queue.append(number)
        self.sum += number
        return self.sum/self.size if len(self.list_queue)==self.size else 0


def get_idx_info(label, n_cls, train_mask):
    '''
    Parameters:
    label: Labels of all nodes  
    n_cls: Total number of classes  
    train_mask: tensor([ True,  True,  True,  ..., False, False, False])  

    Output:
    Node indices for each category
    '''
    label = label.squeeze()
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info


def construct_graph_with_adj(node_logits, adj_logits, real_labels, top_ratio=0.75):
    """
    Construct a graph data structure where nodes of the same category are connected by edges (including self-loops), and extract edge weights from adj_logits
    Parameters:
        node_logits (Tensor): Node feature matrix with shape [num_nodes, num_features]
        adj_logits (Tensor): Edge weight matrix with shape [num_nodes, num_nodes]
        real_labels (Tensor): Ground truth labels of nodes with shape [num_nodes]

    Returns:
        Data: PyG Data object containing node features, edge indices, and edge weights
    """

    num_nodes = real_labels.size(0)

    # Flatten and sort (ignore diagonal elements as self-loops are handled separately).
    non_diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=adj_logits.device)
    non_diag_values = adj_logits[non_diag_mask].flatten()
    # Calculate the index of the top 1/4 position.
    k = max(1, int(len(non_diag_values) * top_ratio))
    edge_thre = torch.kthvalue(non_diag_values, k).values.item()
    
    edge_index = [[], []]
    edge_weight = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                weight = 1.0
            elif real_labels[i] == real_labels[j] and adj_logits[i][j] > edge_thre:
                weight = adj_logits[i][j]
            else:
                weight = 0

            if weight != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_weight.append(weight)

    graph = Data(
        x=node_logits,
        edge_index=torch.tensor(edge_index),
        edge_weight=torch.tensor(edge_weight),
        y=real_labels
    )
    return graph


class DiversityLoss(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        #Compute the pairwise distance matrix between all samples in the input tensor.
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
    

class TrainingLossEarlyStopper:
    def __init__(self, patience=10, warmup_rounds=15, 
                 improvement_thresh=1e-2, variance_thresh=1e-2, improve_window=10, var_window=8):
        self.patience = patience
        self.warmup_rounds = warmup_rounds
        self.improvement_thresh = improvement_thresh
        self.variance_thresh = variance_thresh
        self.loss_history = []
        self.wait_count = 0
        self.best_loss = float('inf')
        self.improve_window = improve_window
        self.var_window = var_window
        
    def should_stop(self, current_loss, round):
        self.loss_history.append(current_loss)
        
        if round < self.warmup_rounds:
            return False
            
        # Condition 2: Recent Improvement Rate
        improvement = self._calculate_improvement()
        # Condition 3: Loss Volatility
        variance = self._check_variance()

        stop_conditions = [improvement < self.improvement_thresh, 
                          variance < self.variance_thresh]
        print("improvement: ", improvement, " self.improvement_thresh: ", self.improvement_thresh, " variance: ", variance, " self.variance_thresh: ", self.variance_thresh)
        print("stop_conditions: ", stop_conditions)
        
        if sum(stop_conditions) >= 2:
            self.wait_count += 1
        else:
            self.wait_count = 0
            
        return self.wait_count >= self.patience
    
    def _calculate_improvement(self):
        if len(self.loss_history) < self.improve_window * 2:
            return float('inf')
        recent = np.mean(self.loss_history[-self.improve_window:])
        previous = np.mean(self.loss_history[-self.improve_window*2:-self.improve_window])
        return (previous - recent) / abs(previous)
    
    def _check_variance(self):
        if len(self.loss_history) < self.var_window:
            return False
        return np.var(self.loss_history[-self.var_window:])

previous_data_cache = {}
def cache_previous_data(worker_idx, new_x, new_edge_index, new_edge_weight, new_y, new_train_idx):
    previous_data_cache[worker_idx] = {
        'new_x': new_x,
        'new_edge_index': new_edge_index,
        'new_edge_weight': new_edge_weight,
        'new_y': new_y,
        'new_train_idx': new_train_idx
    }

def load_previous_data(worker_idx):
    if worker_idx in previous_data_cache:
        return previous_data_cache[worker_idx]['new_x'], \
               previous_data_cache[worker_idx]['new_edge_index'], \
               previous_data_cache[worker_idx]['new_edge_weight'], \
               previous_data_cache[worker_idx]['new_y'], \
               previous_data_cache[worker_idx]['new_train_idx']
    else:
        print(f"No data in cache, worker_idx={worker_idx}")
        return None, None, None, None, None


def main(args, logger):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    Coauthor_list = ["Cs","Physics"]
    Amazon_list = ["computers","photo"]
    ##### DATA PREPARATION #####
    if (args.dataset == 'Cora' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='./data/', \
                            name=args.dataset, \
                            transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Cora-full'):
        dataset = CoraFull(root='./data/',
                           transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Citeseer'):
        dataset = Planetoid(root='./data/', \
                            name=args.dataset)
    elif (args.dataset == 'Flickr'):
        dataset = Flickr(root='./data/Flickr/', \
                         transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Reddit2'):
        dataset = Reddit2(root='./data/Reddit2/', \
                          transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Reddit'):
        dataset = Reddit(root='./data/Reddit', \
                          transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Yelp'):
        dataset = Yelp(root='./data/Yelp/', \
                          transform=T.LargestConnectedComponents())
        labels = np.argmax(dataset.data.y.numpy(), axis=1) + 1
        data = dataset.data
        data.y = torch.from_numpy(labels).reshape(-1, 1)
    elif (args.dataset == 'ogbn-arxiv'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
    elif (args.dataset == 'ogbn-products'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name='ogbn-products', root='./data/')
    elif (args.dataset == 'ogbn-proteins'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root='./data/')
    elif (args.dataset in Coauthor_list):
        dataset = Coauthor(root='./data/',name =args.dataset,  \
                          transform=T.NormalizeFeatures())
        print('datasets', dataset[0])
    elif (args.dataset in Amazon_list):
        dataset = Amazon(root='./data/',name =args.dataset,  \
                          transform=T.LargestConnectedComponents())
    print("The current process ID is:", os.getpid())
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    ogbn_data_list = ["ogbn-arxiv",'ogbn-products','ogbn-proteins']
    if args.dataset in ogbn_data_list:
        data = ogba_data(dataset)
    elif args.dataset in Amazon_list:
        data = Amazon_data(dataset)
        data.y = data.y.to(dtype=torch.long)
    elif args.dataset in Coauthor_list:
        data = Coauthor_data(dataset)
    else:
        data = dataset[0]
    if args.dataset == 'ogbn-proteins':
        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')
        _, f_dim = data.x.size()
        print(f'ogbn-proteins Number of features: {f_dim}')
        print("data.y = data.y.to(torch.float)", data.y.shape)
    if args.dataset == 'Reddit':
        data.y = data.y.long()
    args.avg_degree = data.num_edges / data.num_nodes
    nclass = int(data.y.max() + 1)
    args.class_num = nclass
    print("class", int(data.y.max() + 1))
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f"the feature of node[0]: {dataset[0].x}")
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print('======================Start Splitting the Data to all clients========================================')
    print("split method: ", args.is_iid)
    if args.is_iid == "iid":
        client_data, client_idx = split_Random(args, data)  
    elif args.is_iid == "non-iid-louvain":
        client_data = split_Louvain(args, data)
        print("louvain factor: ", args.louvain_alpha)
    elif args.is_iid == "non-iid-Metis":
        client_data = split_Metis(args, data)
    elif args.is_iid == "non-iid-dirichlet":
        client_data, client_idx = split_dirichlet(args, data)
        print("dirichlet factor: ", args.dirichlet_alpha)
    elif args.is_iid == "non-iid-graph-kernal":
        client_data = split_graph_kernal(args, data)
        print("kernal factor: ", args.kernal_alpha)
    else:
        raise NameError

    for i in range(args.num_workers):
        client_data[i], idx_train, idx_test, origin_train_index, origin_test_indexget_split = get_split(args, client_data[i], device, client_idx[i])
            
    print('======================Start Preparing the Data========================================')
    client_idx_train = []
    client_idx_test = []

    for k in range(args.num_workers):    
        print("Client:{}".format(k))
        print(client_data[k])
        # Gather some statistics about the graph.
        print(f'Number of nodes: {client_data[k].num_nodes}')
        print(f'Number of edges: {client_data[k].num_edges}')

        local_idx_train = client_data[k].train_mask.nonzero(as_tuple=True)[0]
        local_idx_test = client_data[k].test_mask.nonzero(as_tuple=True)[0]
        client_idx_train.append(local_idx_train)
        client_idx_test.append(local_idx_test)
        print(f'local idx train: {local_idx_train}')
        print(f'local idx test: {local_idx_test}')
        client_data[k].edge_index = to_undirected(client_data[k].edge_index)
        edge_weight = torch.ones([client_data[k].edge_index.shape[1]], device=device, dtype=torch.float) #create weight tensor with initial weight 1(num equals edge number)
        client_data[k].edge_weight = edge_weight
        # Normalize the data of each client
        if args.dataset == 'Reddit' or args.dataset == "Cora-full" or args.dataset == "computers":
            pass
        else:
            x_norm = F.normalize(
                client_data[k].x.clone().to(device),
                p=2, 
                dim=1
                )
            client_data[k].x = x_norm
           
    print('======================Start Preparing the Models========================================')
    config_file = f"yaml/{args.num_workers}_heterogeneous_GNNs.yaml"
    if not os.path.exists(config_file):
        raise ValueError(f"The configuration file does not exist: {config_file}")
    with open(config_file, 'r') as f:
        client_configs = yaml.safe_load(f)

    local_model_list = []
    for i in range(args.num_workers):
        client_id = f'client_{i}'
        client_config = client_configs.get(client_id, {})

        if not client_config:
             raise ValueError(f"Client configuration {client_id} does not exist")
        
        model_config = client_config['model']
        model_name = model_config['name']     
        hidden_dim = model_config['hidden']
        dropout = model_config['dropout']
        layer = model_config['layer']
        
        client_model = model_construct(args, model_name, client_data[i], device, nclass, hidden=hidden_dim, dropout=dropout, layer=layer).to(device)  
        local_model_list.append(client_model)


    # Build a generator.
    generator = model_construct(args, "FedKD_Generator", client_data[0], device, nclass).to(device)
    generator_optimizer = Adam(generator.parameters(), lr=0.01, weight_decay=args.weight_decay)
    # Build a discriminator.
    if args.use_GAN:
        discriminator_model_list = []
        for i in range(args.num_workers):
            dis_test_model = model_construct(args, "Discriminator", client_data[0], device, None).to(device)
            discriminator_model_list.append(dis_test_model)
    
    # Generator loss change monitoring
    if args.dataset == 'Cs' or args.dataset == 'Citeseer':
        generator_stopper = TrainingLossEarlyStopper(patience=10, variance_thresh=0.05, improve_window=5, var_window=5)
    else:
        generator_stopper = TrainingLossEarlyStopper(patience=10)
    fixed_pseudo_graph_step = 10000


    print('======================Start Training Model========================================')
    epoch_acc_limit = MoveAvg(size=args.target_round)
    round_reach_target_acc = 0
    prev_out = {}

    c_cnt = [args.sample_num // args.class_num] * args.class_num
    remainder = args.sample_num % args.class_num
    for i in range(remainder):
        c_cnt[i] += 1
    print(f"pseudo label distribution: {c_cnt}")
    label_distribution = torch.zeros(args.sample_num).to(device).long()
    ptr = 0
    for class_i in range(args.class_num):
        for _ in range(c_cnt[class_i]):
            label_distribution[ptr] = class_i
            ptr += 1
    print("Labels of generated pseudo nodes:", label_distribution)

    max_accuracy = 0

    all_class_flag = False
    server_cache = PseudoGraphCache(device=device, max_size=args.cache_size)

    for epoch in range(args.epochs):
        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None}
            worker_results[f"client_{i}"] = {"pseudo_loss": None}
            worker_results[f"client_{i}"] = {"pseudo_acc": None}
            
        #1. Generate public dataset + model aggregation + model distribution.
        if args.cal_communication:
            start_time_dis = time.time()

        generator.eval()

        for _ in range(args.pseudo_num_per_time):
            #1.1 Generate a public dataset.
            if epoch < fixed_pseudo_graph_step:
                z = torch.randn((args.sample_num, args.noise_dim)).to(device)
                node_logits, adj_matrix, z_c = generator.forward(z=z, c=label_distribution)
                pseudo_graph = construct_graph_with_adj(node_logits=node_logits.detach(), adj_logits=adj_matrix.detach(), real_labels=label_distribution, top_ratio=args.top_ratio)
            else:
                pseudo_graph = server_cache.get_last_graph()

            #1.2 Clean the generated public dataset.
            if args.pseudo_graph_clean:
                print("Performing pseudo-data cleaning!")
                sample_loss_list = []
                with torch.no_grad():
                    for j in range(args.num_workers):
                        local_model_list[j].eval()
                        teacher_logits = local_model_list[j].forward_logits(
                            pseudo_graph.x.to(device), 
                            pseudo_graph.edge_index.to(device), 
                            pseudo_graph.edge_weight.to(device)
                        )
                        sample_loss = F.cross_entropy(teacher_logits, pseudo_graph.y.to(device), reduction='none')
                        sample_loss_list.append(sample_loss)
                sample_loss_stack = torch.stack(sample_loss_list, dim=1)    #torch.Size([100, 10])
                if args.clean_method == 'threshold':
                    sample_mean_loss = torch.mean(sample_loss_stack, dim=1)  # torch.Size([100])
                    mean_loss = torch.mean(sample_mean_loss)
                    std_loss = torch.std(sample_mean_loss)
                    loss_threshold = mean_loss + 3 * std_loss
                    mask = sample_mean_loss < loss_threshold
                    keep_node_indices = torch.where(mask)[0]
                elif args.clean_method == 'gmm':
                    sample_loss_stack = sample_loss_stack.cpu().numpy()
                    gmm = GaussianMixture(n_components=2, random_state=args.seed)
                    gmm.fit(sample_loss_stack)
                    clusters = gmm.predict(sample_loss_stack)
                    clusters_tensor = torch.from_numpy(clusters)
                    if gmm.means_[0][0] > gmm.means_[1][0]:
                        clean_cluster = 1
                    else:
                        clean_cluster = 0
                    mask = (clusters_tensor == clean_cluster)
                    keep_node_indices = torch.where(mask)[0]
                print("keep_node_indices: ", keep_node_indices)
                
                edge_index, edge_attr = subgraph(
                    keep_node_indices, 
                    pseudo_graph.edge_index, 
                    edge_attr=pseudo_graph.edge_weight,
                    relabel_nodes=True,
                    num_nodes=pseudo_graph.num_nodes
                )

                clean_pseudo_graph = type(pseudo_graph)(
                    x=pseudo_graph.x[keep_node_indices],
                    y=pseudo_graph.y[keep_node_indices],
                    edge_index=edge_index,
                    edge_weight=edge_attr
                )
                print("Number of original nodes: ", len(pseudo_graph.y), "Number of nodes after filtering: ", len(clean_pseudo_graph.y))
                pseudo_graph = clean_pseudo_graph

            
            #1.3 Store the newly generated data from each round into the cache.
            if args.cache_size > 1 or epoch < fixed_pseudo_graph_step:
                server_cache.add(pseudo_graph)

        #1.4.1 Merge all pseudo-images in the cache, renumber all nodes, and form a large graph.
        if args.cache_size > 1:
            pseudo_graph = server_cache.get_merged_graph()
        #1.4.2 Check if the pseudo_graph contains nodes of all categories.
        unique_classes_in_pseudo = torch.unique(pseudo_graph.y)
        if len(unique_classes_in_pseudo) < args.class_num:
            all_class_flag = False
            print("disappear classes! ", unique_classes_in_pseudo)
        else:
            all_class_flag = True    
        
        #1.5 Model Aggregation: Weighted Multi-Teacher Distillation
        local_class_scores = []
        teacher_weights = []
        weight_strategy = "None"

        for j in range(args.num_workers):
            local_model_list[j].eval()
            teacher_logits = local_model_list[j].forward_logits(
                pseudo_graph.x.to(device), 
                pseudo_graph.edge_index.to(device), 
                pseudo_graph.edge_weight.to(device)
            )
            teacher_logits = nn.functional.softmax(teacher_logits / args.TEMP, dim=1)
            local_class_scores.append(teacher_logits)
            
            # Calculate Teacher Weights
            if weight_strategy == "confidence":
                weight = teacher_logits.max(dim=1)[0].mean()
            elif weight_strategy == "accuracy":
                weight = teacher_val_acc[j]  
            elif weight_strategy == "diversity":
                entropy = -torch.sum(teacher_logits * torch.log(teacher_logits + 1e-8), dim=1)
                weight = 1 / (entropy.mean() + 1e-8)
            else:
                weight = 1.0 
            teacher_weights.append(weight)

        # Normalize the weights.
        teacher_weights = torch.tensor(teacher_weights, device=device)
        teacher_weights = teacher_weights / teacher_weights.sum()

        # Weighted aggregation of teacher outputs.
        weighted_scores = torch.stack(
            [w * s for w, s in zip(teacher_weights, local_class_scores)]
        ).sum(dim=0)

        # 1.5 Student model training via distillation.
        for j in range(args.num_workers):
            loss = local_model_list[j].train_with_logits(
                pseudo_graph.x.to(device),
                pseudo_graph.edge_index.to(device),
                pseudo_graph.edge_weight.to(device),
                weighted_scores.detach().clone(),
                pseudo_graph.y.to(device),
                args, 
                train_iters=10
            )
        print(f"Epoch: {epoch}, Public Loss: {loss}")
        logger.log({"Epoch": epoch, "Public Loss": loss})

        if args.cal_communication:
            end_time_dis = time.time()
            total_time_dis = end_time_dis - start_time_dis
            start_time_train = time.time()
               
        round_overall_performance = []
        round_overall_loss = []
        local_proto_dict = {}
   
        for j in range(args.num_workers):
            #--------------------------------model training section--------------------------------
            #2.0 Perform local data augmentation using the SMOTE algorithm.
            now_idx_train = client_idx_train[j]
            class_num_list = scatter_add(src=torch.ones_like(client_data[j].y[now_idx_train]), index=client_data[j].y[now_idx_train], dim=0)
            idx_info = get_idx_info(client_data[j].y, args.class_num, client_data[j].train_mask)
            if epoch > args.warmup and all_class_flag:
                if epoch < fixed_pseudo_graph_step:
                    prev_out_local = prev_out[j]
                    train_idx_list = client_idx_train[j]
                    local2global = {i:train_idx_list[i].item() for i in range(len(train_idx_list))} 
                    global2local = dict([val, key] for key, val in local2global.items())

                    idx_info_list = [item.cpu().tolist() for item in idx_info] 
                    idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]
                    tau = 2
                    max_flag = True
                    no_mask_flag = False
                    same_class_flag = True
                    sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local, idx_info_local, client_idx_train[j], client_idx_test[j], tau, max_flag, no_mask_flag, same_class_flag, pseudo_graph) 
                    # semimaxup
                    neighbor_dist_list = to_dense_adj(client_data[j].edge_index, max_num_nodes=client_data[j].num_nodes).squeeze(0) # Sparse adjacency matrix -> Dense matrix [num_nodes, num_nodes]
                    neighbor_dist_list.fill_diagonal_(1)
                    new_edge_index = neighbor_sampling(client_data[j].x.size(0), client_data[j].edge_index, sampling_src_idx)
                    beta = torch.distributions.beta.Beta(2, 1)
                    lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
                    new_x = saliency_mixup(client_data[j].x, pseudo_graph.x, sampling_src_idx, sampling_dst_idx, lam)
                    new_edge_weight = torch.ones([new_edge_index.shape[1]], device=device, dtype=torch.float) #create weight tensor with initial weight 1(num equals edge number)
                    _new_y = client_data[j].y[sampling_src_idx].clone()
                    new_y = torch.cat((client_data[j].y, _new_y), dim=0)
                    add_num = new_x.shape[0] - client_data[j].x.shape[0]
                    _new_train_idx = torch.arange(client_data[j].num_nodes, client_data[j].num_nodes+add_num)
                    new_train_idx = torch.cat((client_idx_train[j], _new_train_idx), dim=0)

                    loss_train, local_proto_label, output_logits = local_model_list[j].fit(None, new_x.to(device),
                                                                new_edge_index.to(device),
                                                                new_edge_weight.to(device),
                                                                new_y.to(device),
                                                                new_train_idx.to(device),
                                                                {},
                                                                args,
                                                                None,
                                                                train_iters=args.inner_epochs,
                                                                verbose=False)
                    cache_previous_data(j, new_x, new_edge_index, new_edge_weight, new_y, new_train_idx)

                else:
                    new_x, new_edge_index, new_edge_weight, new_y, new_train_idx = load_previous_data(j)
                    if new_x is None:
                        new_x = client_data[j].x
                        new_edge_index = client_data[j].edge_index
                        new_edge_weight = torch.ones([new_edge_index.shape[1]], device=device)
                        new_y = client_data[j].y
                        new_train_idx = client_idx_train[j]
                    loss_train, local_proto_label, output_logits = local_model_list[j].fit(None, new_x.to(device),
                                                                new_edge_index.to(device),
                                                                new_edge_weight.to(device),
                                                                new_y.to(device),
                                                                new_train_idx.to(device),
                                                                {},
                                                                args,
                                                                None,
                                                                train_iters=args.inner_epochs,
                                                                verbose=False)
            else:
                loss_train, local_proto_label, output_logits = local_model_list[j].fit(None,client_data[j].x.to(device),
                                                client_data[j].edge_index.to(device),
                                                client_data[j].edge_weight.to(device),
                                                client_data[j].y.to(device),
                                                client_idx_train[j].to(device),
                                                {},
                                                args,
                                                None,
                                                train_iters=args.inner_epochs,
                                                verbose=False)
            prev_out[j] = output_logits[:client_data[j].num_nodes].detach().clone()
            
            print("Client: {} ,Loss train: {:.4f}".format(j, loss_train))
            
            round_overall_loss.append(loss_train)
            agg_local_proto = agg_local_proto_func(local_proto_label)
            local_proto_dict[j] = agg_local_proto
            
            # save worker results
            for ele in worker_results[f"client_{j}"]:
                if ele == "train_loss":
                    worker_results[f"client_{j}"][ele] = loss_train
            
            #2.2 Train the local discriminator model (with generator fixed).
            if args.use_GAN:
                loss = discriminator_model_list[j].train_step(client_data[j], device, pseudo_graph)
                print("Client: {}, Loss discriminator: {:.4f}".format(j, loss))
        
        if args.cal_communication:
            end_time_train = time.time()
            total_time_train = end_time_train - start_time_train
            
        args.num_selected_models = args.num_workers

        # No aggregation

        #visualize node embeddings
        if args.draw_decision_bound:
            if epoch > args.warmup and all_class_flag:
                visualize_node_embeddings_new_data(
                    local_model=local_model_list[j],
                    new_x=new_x,
                    new_edge_index=new_edge_index,
                    new_edge_weight=new_edge_weight,
                    _new_train_idx=_new_train_idx,
                    new_y=new_y,
                    client_idx_train=client_idx_train[j],
                    client_idx_test=client_idx_test[j],
                    pseudo_graph=pseudo_graph,
                    device=device,
                    class_num=args.class_num,
                    client_id=j,
                    epoch=epoch,
                    output_dir=f"plot_figures/{args.dataset}",
                    perplexity=20,
                    learning_rate=50,
                    n_iter=2000
                )
            else:
                visualize_node_embeddings(
                    local_model=local_model_list[j],
                    client_data=client_data[j],
                    client_idx_train=client_idx_train[j],
                    client_idx_test=client_idx_test[j],
                    pseudo_graph=pseudo_graph,
                    device=device,
                    class_num=args.class_num,
                    client_id=j,
                    epoch=epoch,
                    output_dir=f"plot_figures/{args.dataset}",
                    perplexity=20,
                    learning_rate=50,
                    n_iter=2000
                )

        # global results after agg 
        client_acc_list = []
        for c in range(args.num_workers):
            #load parameters of local model
            acc_test_client = local_model_list[c].test(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
            client_acc_list.append(acc_test_client)
        acc_global = sum(client_acc_list)/len(client_acc_list)

        round_overall_performance.append(acc_global)
        round_average_overall_loss = np.array(round_overall_loss).sum() / args.num_workers

        if epoch_acc_limit.add_num(acc_global) > args.target_acc and round_reach_target_acc == 0:
            round_reach_target_acc = epoch
        
        if acc_global > max_accuracy:
            max_accuracy = acc_global

        print("Round: {}: Average Performance of all clients on clean test set: {:.4f}".format(epoch, acc_global))
        logger.log({"Round": epoch, "Round Average Accuracy": acc_global, "Round Average Loss": round_average_overall_loss, "Max Accuracy": max_accuracy})

        if args.cal_communication:
            start_time_train_gen = time.time()

        # train generator 
        if epoch < fixed_pseudo_graph_step:
            loss_generator = 0
            for _ in range(10):
                generator.train()    
                loss_sem = 0.0
                loss_div = 0.0
                loss_real = 0.0
                generator_optimizer.zero_grad()
                z = torch.randn((args.sample_num, args.noise_dim)).to(device)
                node_logits, adj_matrix, z_c = generator.forward(z=z, c=label_distribution)
                pseudo_graph = construct_graph_with_adj(node_logits=node_logits, adj_logits=adj_matrix, real_labels=label_distribution, top_ratio=args.top_ratio)
                for client_id in range(args.num_workers):
                    local_model_list[client_id].eval()
                    local_pred, local_proto, local_logits = local_model_list[client_id].forward(pseudo_graph.x.to(device), pseudo_graph.edge_index.to(device), None)
                    loss_sem += 1/(args.num_workers) * nn.CrossEntropyLoss()(local_pred, label_distribution)

                loss_div = ContrastiveDiversityLoss(temperature=0.1, metric='cosine').to(device)(local_proto, z)

                loss_G = args.lam1 * loss_sem + args.lam2 * loss_div + args.lam_real * loss_real
                
                loss_G.backward()
                generator_optimizer.step()
                loss_generator += loss_G.item()
                
                print("Generator loss: loss_sem:{} loss_div:{} loss_real:{}".format(loss_sem, loss_div, loss_real))

            logger.log({"loss_G": loss_generator, "epoch": epoch})
            if generator_stopper.should_stop(loss_generator, epoch):
                print("Generator training early stopped at epoch {}".format(epoch))
                fixed_pseudo_graph_step = epoch
                with open("fedkd_early_stop_epoch_log.txt", "a") as f:
                    f.write("dataset {} clients_num {} fedkd_low_cost early stop epoch {}\n".format(
                        args.dataset, args.num_workers, epoch))
        
        if args.cal_communication:
            end_time_train_gen = time.time()
            total_time_train_gen = end_time_train_gen - start_time_train_gen
            total_time = total_time_dis + total_time_train + total_time_train_gen
            print(f"per_round_dis_time (s): {total_time_dis:.6f} s")
            print(f"per_round_dis_time (h): {total_time_dis/(60*60):.6f} hours")
            print(f"per_round_train_time (s): {total_time_train:.6f} s")
            print(f"per_round_train_time (h): {total_time_train/(60*60):.6f} hours")
            print(f"per_round_train_gen (s): {total_time_train_gen:.6f} s")
            print(f"per_round_train_gen (h): {total_time_train_gen/(60*60):.6f} hours")
            print(f"total_time (s): {total_time:.6f} s")
            print(f"total_time (h): {total_time/(60*60):.6f} hours")

    f1_score_end_list = []
    client_accs_list = []
    for c in range(args.num_workers):
        client_f1_score = local_model_list[c].cal_f1_score(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
        f1_score_end_list.append(client_f1_score)
        each_class_acc = local_model_list[c].test_each_class(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
        print("each_class_acc: ", each_class_acc)
        client_accs_list.append(each_class_acc)
    
    cls_acc_avg = avg_per_class_acc(client_accs_list, args.class_num)
    cls_avg_df = pd.DataFrame(cls_acc_avg, index=['accuracy'])
    print(cls_avg_df)
    folder_path = f'script/csv/alpha_{args.dirichlet_alpha}/{args.dataset}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    cls_avg_df.to_csv(f'{folder_path}/{args.alg_method}_{args.dirichlet_alpha}_{args.dataset}.csv')
   
    f1_score_end = sum(f1_score_end_list)/len(f1_score_end_list)

    round_reach_target_acc = args.epochs if round_reach_target_acc == 0 else round_reach_target_acc
                
    return acc_global, round_reach_target_acc, max_accuracy, f1_score_end

if __name__ == '__main__':
    main()