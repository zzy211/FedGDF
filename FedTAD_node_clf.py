import time
import torch
import random
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Reddit,Yelp
from torch_geometric.datasets import Coauthor, Amazon
from Node_level_Models.helpers.func_utils import  get_split, get_total_size, agg_local_proto_func, agg_global_proto_func, visualize_node_embeddings, avg_per_class_acc
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
import yaml
import copy

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

def random_walk_with_matrix(T, walk_length, start):
    current_node = start
    walk = [current_node]
    for _ in range(walk_length - 1):
        probabilities = F.softmax(T[current_node, :], dim=0)
        probabilities /= torch.sum(probabilities)
        next_node = torch.multinomial(probabilities, 1).item()
        walk.append(next_node)
        current_node = next_node
    return walk

#Compute the topological embedding of a graph
def cal_topo_emb(edge_index, num_nodes, max_walk_length):
    A = to_dense_adj(add_self_loops(edge_index)[0], max_num_nodes=num_nodes).squeeze()
    D = torch.diag(torch.sum(A, dim=1))
    T = A * torch.pinverse(D)
    result_each_length = []
    
    for i in range(1, max_walk_length+1):    
        result_per_node = []
        for start in range(num_nodes):
            result_walk = random_walk_with_matrix(T, i, start)
            result_per_node.append(torch.tensor(result_walk).view(1,-1))
        result_each_length.append(torch.vstack(result_per_node))
    topo_emb = torch.hstack(result_each_length)
    return topo_emb    

def construct_graph(node_logits, adj_logits, real_labels=None, k=5):
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = add_self_loops(edge_index)[0]
    edge_weight = torch.ones([edge_index.shape[1]], device=node_logits.device, dtype=torch.float) #create weight tensor with initial weight 1(num equals edge number)
    data = Data(x=node_logits, edge_index=edge_index, edge_weight=edge_weight, y=real_labels)
    return data  

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
        edge_weight = torch.ones([client_data[k].edge_index.shape[1]], device=device, dtype=torch.float) #create weight tensor with initial weight 1(num equals edge number)
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

    generator = model_construct(args, "FedTAD_ConGenerator", client_data[0], device, nclass).to(device)
    generator_optimizer = Adam(generator.parameters(), lr=0.01, weight_decay=args.weight_decay)

    #Calculate confidence
    ckr = torch.zeros((args.num_workers, args.class_num)).to(device)
    for client_id in range(args.num_workers):
        local_data = client_data[client_id].to(device)
        graph_emb = cal_topo_emb(edge_index=local_data.edge_index, num_nodes=local_data.x.shape[0], max_walk_length=5).to(device)
        ft_emb = torch.cat((local_data.x, graph_emb), dim=1).to(device)
        for train_i in client_idx_train[client_id]:
            neighbor = local_data.edge_index[1,:][local_data.edge_index[0, :] == train_i] 
            node_all = 0
            for neighbor_j in neighbor:
                node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                node_all += node_kr
            node_all += 1
            node_all /= (neighbor.shape[0] + 1)
            label = local_data.y[train_i]
            ckr[client_id, label] += node_all

    normalized_ckr = ckr / ckr.sum(0)

    print('======================Start Training Model========================================')
    epoch_acc_limit = MoveAvg(size=args.target_round)
    round_reach_target_acc = 0
    
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
    
    each_class_idx = {}
    for class_i in range(args.class_num):
        each_class_idx[class_i] = label_distribution == class_i
        each_class_idx[class_i] = each_class_idx[class_i].to(device)

    max_accuracy = 0
    for epoch in range(args.epochs):
        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None}
            worker_results[f"client_{i}"] = {"link_train_loss": None}
            worker_results[f"client_{i}"] = {"pseudo_acc": None}
                
        round_overall_performance = []
        round_overall_loss = []

        if args.cal_time:
            dis_start_time = time.time()
        # global model broadcast
        #1. Train the local model with generator data.
        generator.eval()
        #1.1 generator forward
        z = torch.randn((args.num_gen, args.noise_dim)).to(device)
        node_logits = generator.forward(z=z, c=label_distribution)
        node_norm = F.normalize(node_logits, p=2, dim=1)
        adj_logits = torch.mm(node_norm, node_norm.t())

        pseudo_graph = construct_graph(node_logits.detach(), adj_logits.detach(), k=args.topk, real_labels=copy.deepcopy(label_distribution))

        local_class_scores = []
        teacher_weights = []

        for j in range(args.num_workers):
            local_model_list[j].eval()
            teacher_logits = local_model_list[j].forward_logits(
                pseudo_graph.x.to(device), 
                pseudo_graph.edge_index.to(device),
                pseudo_graph.edge_weight.to(device)
            )
            teacher_logits = nn.functional.softmax(teacher_logits / args.TEMP, dim=1)
            local_class_scores.append(teacher_logits)
            weight = 1.0
            teacher_weights.append(weight)

        teacher_weights = torch.tensor(teacher_weights, device=device)
        teacher_weights = teacher_weights / teacher_weights.sum()

        weighted_scores = torch.stack(
            [w * s for w, s in zip(teacher_weights, local_class_scores)]
        ).sum(dim=0)

        # 1.3 Model Distribution: Student Model Training
        for j in range(args.num_workers):
            loss = local_model_list[j].train_with_logits(
                pseudo_graph.x.to(device),
                pseudo_graph.edge_index.to(device),
                pseudo_graph.edge_weight.to(device),
                weighted_scores.detach().clone(),  # use weighted output
                pseudo_graph.y.to(device), # torch.tensor(copy.deepcopy(label_distribution)).to(device)
                args, 
                train_iters=10
            )
        # print(f"Epoch: {epoch}, Public Loss: {loss}, Weights: {teacher_weights.cpu().numpy().round(3)}, weighted_scores: {weighted_scores}")
        logger.log({"Epoch": epoch, "Public Loss": loss})
        if args.cal_time:
            dis_end_time = time.time()
            print(f"Time taken for distillation in epoch {epoch}: {dis_end_time - dis_start_time:.2f} seconds")
            train_start_time = time.time()
        
        # local train
        for j in range(args.num_workers):
            #--------------------------------Model Training Part--------------------------------
            loss_train, local_proto_label = local_model_list[j].fit(None,client_data[j].x.to(device),
                                            client_data[j].edge_index.to(device),
                                            client_data[j].edge_weight.to(device),
                                            client_data[j].y.to(device),
                                            client_idx_train[j].to(device),
                                            {},
                                            args,
                                            None,
                                            train_iters=args.inner_epochs,
                                            verbose=False)
            
            print("Client: {} ,Loss train: {:.4f}".format(j, loss_train))
            
            round_overall_loss.append(loss_train)
            
            # save worker results
            for ele in worker_results[f"client_{j}"]:
                if ele == "train_loss":
                    worker_results[f"client_{j}"][ele] = loss_train
            
            # wandb logger
            logger.log(worker_results)
        
        if args.cal_time:
            train_end_time = time.time()
            print(f"Time taken for local training in epoch {epoch}: {train_end_time - train_start_time:.2f} seconds")
        
        # global eval
        client_acc_list = []
        for c in range(args.num_workers):
            #load parameters of local model
            acc_test_client = local_model_list[c].test(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
            client_acc_list.append(acc_test_client)
        acc_global = sum(client_acc_list)/len(client_acc_list)

        round_overall_performance.append(acc_global)
        round_average_overall_loss = np.array(round_overall_loss).sum() / args.num_workers

       # visualize node embeddings
        if args.draw_decision_bound:
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
                output_dir=f"plot_figures/fedtad/{args.dataset}",
                perplexity=20,
                learning_rate=50,
                n_iter=2000
            )
        
        if epoch_acc_limit.add_num(acc_global) > args.target_acc and round_reach_target_acc == 0:
            round_reach_target_acc = epoch

        if acc_global > max_accuracy:
            max_accuracy = acc_global

        print("Round: {}: Average Performance of all clients on clean test set: {:.4f}".format(epoch, acc_global))
        logger.log({"Round": epoch, "Round Average Accuracy": acc_global, "Round Average Loss": round_average_overall_loss, "Max Accuracy": max_accuracy})

        each_class_idx = {}
        for class_i in range(args.class_num):
            each_class_idx[class_i] = (label_distribution == class_i)

        if args.cal_time:
            train_gen_start_time = time.time()
        for _ in range(10):
            generator.train()
            loss_sem = 0
            loss_div = 0
            generator_optimizer.zero_grad()
            z = torch.randn((args.num_gen, args.noise_dim)).to(device)
            node_logits = generator.forward(z=z, c=label_distribution)
            node_norm = F.normalize(node_logits, p=2, dim=1)
            adj_logits = torch.mm(node_norm, node_norm.t())
            pseudo_graph = construct_graph(node_logits, adj_logits, real_labels=label_distribution, k=args.topk)
            for client_id in range(args.num_workers):
                #generator forward             
                #local & global model --> forward
                local_model_list[client_id].eval()
                local_pred, proto, output_logits = local_model_list[client_id].forward(pseudo_graph.x.to(device), pseudo_graph.edge_index.to(device), None)

                # loss_sem += 1/(args.num_workers) * nn.CrossEntropyLoss()(local_pred, label_distribution)
                for class_i in range(args.class_num):
                    loss_sem += normalized_ckr[client_id][class_i] * nn.CrossEntropyLoss()(local_pred[each_class_idx[class_i]], label_distribution[each_class_idx[class_i]])

            #diversity loss
            loss_div += DiversityLoss(metric='l2').to(device)(z.view(z.shape[0],-1), node_logits) 
            loss_G = args.lam1 * loss_sem + args.lam2 * loss_div
            print("loss_sem: ", loss_sem.item(), " loss_div:", loss_div.item())
            loss_G.backward()
            generator_optimizer.step()
        if args.cal_time:
            train_gen_end_time = time.time()
            print(f"Time taken for generator training in epoch {epoch}: {train_gen_end_time - train_gen_start_time:.2f} seconds")

    
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