import torch
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Reddit,Yelp
from torch_geometric.datasets import Coauthor, Amazon
# import Node_level_Models.helpers.selection_utils  as hs
from Node_level_Models.helpers.func_utils import  get_split, get_total_size, agg_local_proto_func, agg_global_proto_func, avg_per_class_acc
from Node_level_Models.helpers.split_graph_utils import split_Random, split_Louvain, split_Metis, split_dirichlet,split_graph_kernal
from Node_level_Models.models.construct import model_construct
from Node_level_Models.data.datasets import  ogba_data,Amazon_data,Coauthor_data
from Node_level_Models.aggregators.aggregation import fed_avg, fed_cls
from collections import deque
import pandas as pd
import os
import copy
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, subgraph, k_hop_subgraph
import yaml
from sklearn.cluster import AgglomerativeClustering
from torch_scatter import scatter_add
from Node_level_Models.helpers.gens_sha import sampling_node_source, neighbor_sampling, saliency_mixup
from torch_geometric.utils import to_dense_adj
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
    Node indices for each class
    '''
    label = label.squeeze()
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info

def augmentation_data_func(client_data, local_logits, sample_num=100):
    #output: augmentation_data
    softmax_logits = F.softmax(local_logits, dim=1)
    node_mean_logits = torch.mean(softmax_logits, dim=0, keepdim=True)
    samples = torch.multinomial(node_mean_logits.squeeze(), sample_num, replacement=True) #ouput tensors
    
    #new_features, new_edges
    new_feature = []
    new_label = []
    new_edge = [[], []]
    new_id = client_data.num_nodes
    
    for node in samples:
        label = node.item()
        probabilities = softmax_logits[:, label].squeeze()
        probabilities = F.softmax(probabilities, dim=0)
        source_node = torch.multinomial(probabilities, 1, replacement=True).to('cpu') #tensor
        #Find the target node from a neighboring node.
        num_hops = 2
        neighbors, _, _, _ = k_hop_subgraph(torch.tensor([source_node]), num_hops, client_data.edge_index, num_nodes=client_data.num_nodes)
        same_label_pro = softmax_logits[neighbors, label]
        same_label_pro = F.softmax(same_label_pro, dim=0)
        
        if(same_label_pro.numel()!=0):
            target_node_id = torch.multinomial(same_label_pro, 1, replacement=True).to('cpu')
            target_node = neighbors[target_node_id]
            l1 = 0.7
            pseudo_node_feature = client_data.x[source_node] * l1 + client_data.x[target_node] * (1 - l1)
            new_feature.append(pseudo_node_feature.squeeze())
            new_label.append(0)
            new_edge[0].append(source_node.item())
            new_edge[1].append(new_id)
            new_edge[0].append(new_id)
            new_edge[1].append(target_node.item())
            new_id += 1

    old_features = client_data.x
    new_features = torch.stack(new_feature,dim=0)
    output_features = torch.cat([old_features, new_features], dim=0)

    old_labels = client_data.y
    output_labels = torch.cat([old_labels, torch.tensor(new_label)], dim=0)

    old_edge_index = client_data.edge_index #(2, n)
    new_edge_index = new_edge   #(2, m)
    output_edge_index = torch.cat([old_edge_index, torch.tensor(new_edge_index)], dim=1)    #(2, n+m)
    output_edge_index = to_undirected(output_edge_index)    #transform into undirected graph

    new_node_num = new_features.shape[0]

    old_train_mask = client_data.train_mask
    new_train_mask = torch.zeros(new_node_num, dtype=torch.bool)
    train_mask = torch.cat([old_train_mask, new_train_mask], dim=0)

    total_node_num = train_mask.shape[0]

    old_test_mask = client_data.test_mask
    new_test_mask = torch.zeros(new_node_num, dtype=torch.bool)
    test_mask = torch.cat([old_test_mask, new_test_mask], dim=0)
 
    restored_data = Data(
        x=output_features,
        edge_index=torch.tensor(output_edge_index),
        edge_attr=None,
        y=output_labels,
        train_mask=train_mask,
        val_mask=torch.zeros(total_node_num, dtype=torch.bool),
        test_mask=test_mask
    )
    return restored_data

import torch
import torch.nn.functional as F
import numpy as np

def agg_customized_global_proto_func(local_proto_dict, num_workers):
    """
    Aggregate client prototypes to generate global prototypes (including FINCH clustering)
    Args:
        local_proto_dict: dict, key is client_id, value is client_proto_dict
        num_workers: int, number of clients

    Returns:
        dict: key is client_id, value is the corresponding global prototype dictionary
    """
    # Step 1: Aggregate prototypes from all clients
    agg_global_proto = {}
    for client_id, local_proto in local_proto_dict.items():
        for label, proto in local_proto.items():
            if label not in agg_global_proto:
                agg_global_proto[label] = [proto]
            else:
                agg_global_proto[label].append(proto)
    # Step 2: Perform FINCH clustering on prototypes of each label
    clustered_global_proto = {}
    for label, proto_list in agg_global_proto.items():
        proto_array = torch.stack(proto_list).detach().cpu().numpy()
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
        cluster_labels = clustering.fit_predict(proto_array)
        unique_clusters = np.unique(cluster_labels)
        cluster_centers = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_protos = proto_array[cluster_mask]
            cluster_center = np.mean(cluster_protos, axis=0)
            cluster_centers.append(torch.tensor(cluster_center, device=proto_list[0].device))
        clustered_global_proto[label] = cluster_centers
    # Step 3: Generate personalized global prototypes for each client
    client_agg_proto = {}
    for client_id in range(num_workers):
        global_proto_dict = {}
        local_client_proto = local_proto_dict[client_id]
        for label, local_proto in local_client_proto.items():
            if label not in clustered_global_proto:
                cluster_centers = agg_global_proto[label]
            else:
                cluster_centers = clustered_global_proto[label]
            similarity_list = []
            for center in cluster_centers:
                cos_similarity = F.cosine_similarity(center, local_proto, dim=0)
                similarity_list.append(cos_similarity)
            max_index = max(enumerate(similarity_list), key=lambda x: x[1])[0]
            num_centers = len(similarity_list)
            l_mu = 0.7
            if num_centers > 1:
                weights = [(1 - l_mu) / (num_centers - 1)] * num_centers
                weights[max_index] = l_mu
            else:
                weights = [1.0]
            weighted_global_proto = torch.zeros_like(cluster_centers[0])
            for j, weight in enumerate(weights):
                weighted_global_proto += weight * cluster_centers[j]
            
            global_proto_dict[label] = weighted_global_proto
        client_agg_proto[client_id] = global_proto_dict
    return client_agg_proto

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
        dataset = Reddit(root='./data/Reddit/', \
                          transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Yelp'):
        dataset = Yelp(root='./data/Yelp/', \
                          transform=T.LargestConnectedComponents())
        # Convert one-hot encoded labels to integer labels
        labels = np.argmax(dataset.data.y.numpy(), axis=1) + 1
        # Create new data object with integer labels
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
        data = dataset[0]  # Get the graph object.
    if args.dataset == 'ogbn-proteins':
        # Initialize features of nodes by aggregating edge features.
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
        edge_weight = torch.ones([client_data[k].edge_index.shape[1]], device=device, dtype=torch.float)
        client_data[k].edge_weight = edge_weight
        
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

    print('======================Start Training Model========================================')
    epoch_acc_limit = MoveAvg(size=args.target_round)
    round_reach_target_acc = 0
    client_agg_proto = {i: {} for i in range(0, args.epochs)}
    max_accuracy = 0
    prev_out = {}   #key is client_id, value is the predicted probabilities of all samples for this client
    
    for epoch in range(args.epochs):
        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None}
            worker_results[f"client_{i}"] = {"link_train_loss": None}    
                
        round_overall_performance = []
        round_overall_loss = []
        local_proto_dict = {}
        
        if args.cal_time:
            train_start_time = time.time()
        for j in range(args.num_workers):
            #--------------------------------Data Augmentation for Each Client First--------------------------
            if epoch > 1:
                now_idx_train = client_idx_train[j]
                class_num_list = scatter_add(src=torch.ones_like(client_data[j].y[now_idx_train]), index=client_data[j].y[now_idx_train], dim=0)
                idx_info = get_idx_info(client_data[j].y, args.class_num, client_data[j].train_mask)
                prev_out_local = prev_out[j]
                train_idx_list = client_idx_train[j]
                local2global = {i:train_idx_list[i].item() for i in range(len(train_idx_list))}
                global2local = dict([val, key] for key, val in local2global.items())

                idx_info_list = [item.cpu().tolist() for item in idx_info] 
                idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]
                tau = 2
                max_flag = False
                no_mask_flag = False
                same_class_flag = True  # Source node and target node belong to the same class
                sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local, idx_info_local, client_idx_train[j], tau, max_flag, no_mask_flag, same_class_flag) 
                # semimaxup
                neighbor_dist_list = to_dense_adj(client_data[j].edge_index, max_num_nodes=client_data[j].num_nodes).squeeze(0) # Sparse adjacency matrix -> Dense matrix [num_nodes, num_nodes]
                neighbor_dist_list.fill_diagonal_(1)  # Set diagonal elements to 1
                new_edge_index = neighbor_sampling(client_data[j].x.size(0), client_data[j].edge_index, sampling_src_idx, neighbor_dist_list)
                beta = torch.distributions.beta.Beta(10, 1)
                lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
                new_x = saliency_mixup(client_data[j].x, sampling_src_idx, sampling_dst_idx, lam)
                new_edge_weight = torch.ones([new_edge_index.shape[1]], device=device, dtype=torch.float) #create weight tensor with initial weight 1(num equals edge number)

                #create new_dataset
                _new_y = client_data[j].y[sampling_src_idx].clone()
                new_y = torch.cat((client_data[j].y, _new_y), dim=0)
                add_num = new_x.shape[0] - client_data[j].x.shape[0]
                _new_train_idx = torch.arange(client_data[j].num_nodes, client_data[j].num_nodes+add_num)
                new_train_idx = torch.cat((client_idx_train[j], _new_train_idx), dim=0)

                #--------------------------------Model Training Part-----------------------------
                loss_train, local_proto_label, output_logits = local_model_list[j].fit(None, new_x.to(device),
                                                new_edge_index.to(device),
                                                new_edge_weight.to(device),
                                                new_y.to(device),
                                                new_train_idx.to(device),
                                                client_agg_proto[j],
                                                args,
                                                None,
                                                train_iters=args.inner_epochs,
                                                verbose=False)
                prev_out[j] = output_logits[:client_data[j].num_nodes].detach().clone()

            else:
                loss_train, local_proto_label, output_logits = local_model_list[j].fit(None, client_data[j].x.to(device),
                                                client_data[j].edge_index.to(device),
                                                client_data[j].edge_weight.to(device),
                                                client_data[j].y.to(device),
                                                client_idx_train[j].to(device),
                                                client_agg_proto[j],
                                                args,
                                                None,
                                                train_iters=args.inner_epochs,
                                                verbose=False)
                prev_out[j] = output_logits.detach().clone()

            print("Client: {} ,Loss train: {:.4f}".format(j, loss_train))

            round_overall_loss.append(loss_train)
            agg_local_proto = agg_local_proto_func(local_proto_label)
            local_proto_dict[j] = agg_local_proto
            
            # save worker results
            for ele in worker_results[f"client_{j}"]:
                if ele == "train_loss":
                    worker_results[f"client_{j}"][ele] = loss_train
            
            # wandb logger
            logger.log(worker_results)
        
        if args.cal_time:
            train_end_time = time.time()
            print(f"Training time for round {epoch}: {train_end_time - train_start_time:.2f} seconds")
            agg_start_time = time.time()

        args.num_selected_models = args.num_workers
               
        # Aggregation
        client_agg_proto = agg_customized_global_proto_func(local_proto_dict, args.num_workers)
        if args.cal_time:
            agg_end_time = time.time()
            print(f"Aggregation time for round {epoch}: {agg_end_time - agg_start_time:.2f} seconds")
            total_time = agg_end_time - train_start_time
            print(f"Total time for round {epoch}: {total_time:.2f} seconds")
       
        # global results after aggregation: Accuracy calculated on all clients' test sets using the global model
        client_acc_list = []
        for c in range(args.num_workers):
            #load parameters of local model
            acc_test_client = local_model_list[c].test(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
            client_acc_list.append(acc_test_client)
        acc_global = sum(client_acc_list)/len(client_acc_list)

        if acc_global > max_accuracy:
            max_accuracy = acc_global

        round_overall_performance.append(acc_global)
        round_average_overall_loss = np.array(round_overall_loss).sum() / args.num_workers

        if epoch_acc_limit.add_num(acc_global) > args.target_acc and round_reach_target_acc == 0:
            round_reach_target_acc = epoch

        print("Round: {}: Average Performance of all clients on clean test set: {:.4f}".format(epoch, acc_global))
        logger.log({"Round": epoch, "Round Average Accuracy": acc_global, "Round Average Loss": round_average_overall_loss, "Max Accuracy": max_accuracy})
     
    f1_score_end_list = []
    acc_end_list = []
    client_accs_list = []
    for c in range(args.num_workers):
        client_f1_score = local_model_list[c].cal_f1_score(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
        f1_score_end_list.append(client_f1_score)
        each_class_acc = local_model_list[c].test_each_class(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
        print("each_class_acc: ", each_class_acc)
        client_accs_list.append(each_class_acc)
        client_acc = local_model_list[c].test(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
        acc_end_list.append(client_acc)
    
    cls_acc_avg = avg_per_class_acc(client_accs_list, args.class_num)
    cls_avg_df = pd.DataFrame(cls_acc_avg, index=['accuracy'])
    print(cls_avg_df)
    folder_path = f'script/csv/alpha_{args.dirichlet_alpha}/{args.dataset}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    cls_avg_df.to_csv(f'{folder_path}/{args.alg_method}_{args.dirichlet_alpha}_{args.dataset}.csv')
   
    f1_score_end = sum(f1_score_end_list)/len(f1_score_end_list)
        
    acc_global_end = sum(acc_end_list)/len(acc_end_list)

    print("Performance of all clients on clean test set: {:.4f}".format(acc_global_end))
    round_reach_target_acc = args.epochs if round_reach_target_acc == 0 else round_reach_target_acc

    return acc_global_end, round_reach_target_acc, max_accuracy, f1_score_end


if __name__ == '__main__':
    main()