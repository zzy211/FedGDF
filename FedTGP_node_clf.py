import torch
import random
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Reddit,Yelp
from torch_geometric.datasets import Coauthor, Amazon
# import Node_level_Models.helpers.selection_utils  as hs
from Node_level_Models.helpers.func_utils import  get_split, get_total_size, agg_local_proto_func, agg_global_proto_func, avg_per_class_acc
from torch_geometric.utils import to_undirected
from Node_level_Models.helpers.split_graph_utils import split_Random, split_Louvain, split_Metis, split_dirichlet, split_graph_kernal
from Node_level_Models.models.construct import model_construct
from Node_level_Models.data.datasets import ogba_data, Amazon_data, Coauthor_data
from Node_level_Models.aggregators.aggregation import fed_avg, fed_cls
from collections import deque
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml
from torch_geometric.datasets import CoraFull
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
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

class Trainable_Global_Prototypes(nn.Module):
    '''Generate the feature prototype for the class based on the class ID'''
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)
        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out

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
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
    elif (args.dataset == 'ogbn-products'):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-products', root='./data/')
    elif (args.dataset == 'ogbn-proteins'):
        from ogb.nodeproppred import PygNodePropPredDataset
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
    
    #Initialize the prototype generator
    TGP = Trainable_Global_Prototypes(num_classes=args.class_num, 
                                      server_hidden_dim=256,
                                      feature_dim=args.hidden,
                                      device=device).to(device)
    TGP_optimizer = torch.optim.SGD(TGP.parameters(), lr=0.01)

    #print("+++++++++++++ Federated Node Classification +++++++++++++")
    print('======================Start Training Model========================================')
    epoch_acc_limit = MoveAvg(size=args.target_round)
    round_reach_target_acc = 0
    agg_global_proto = {}
    max_accuracy = 0

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
            #-------------------------------- Model Training Part --------------------------------
            loss_train, local_proto_label = local_model_list[j].fit(None,client_data[j].x.to(device),
                                            client_data[j].edge_index.to(device),
                                            client_data[j].edge_weight.to(device),
                                            client_data[j].y.to(device),
                                            client_idx_train[j].to(device),
                                            agg_global_proto,
                                            args,
                                            None,
                                            train_iters=args.inner_epochs,
                                            verbose=False)
            
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
        
        args.num_selected_models = args.num_workers
        if args.cal_time:
            train_end_time = time.time()
            print(f"Time taken for local training in epoch {epoch}: {train_end_time - train_start_time:.2f} seconds")
            agg_start_time = time.time()
               
        # Aggregation
        TGP.eval()
        agg_global_proto_list = TGP(list(range(args.class_num)))
        for cls in range(args.class_num):
            agg_global_proto[cls] = agg_global_proto_list[cls].clone().detach()
        
        # uploaded_protos
        uploaded_protos = []
        for key, local_proto in local_proto_dict.items():
            for label, proto in local_proto.items():
                uploaded_protos.append((proto, label))
        
        # Calculate the minimum distance between classes
        gap = torch.ones(args.class_num, device=device) * 1e9
        avg_protos = agg_global_proto_func(local_proto_dict)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    gap[k1] = torch.min(gap[k1], dis)
                    gap[k2] = torch.min(gap[k2], dis)
        
        # Handle outliers and compute global minimum/maximum inter-class distances
        max_gap = torch.max(gap)

        if args.cal_time:
            agg_end_time = time.time()
            print(f"Time taken for aggregation in epoch {epoch}: {agg_end_time - agg_start_time:.2f} seconds")

        if args.cal_time:
            train_tgp_start_time = time.time()

        # Train the TGP model
        TGP.train()
        for e in range(10):
            proto_loader = DataLoader(uploaded_protos, batch_size=32, 
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(device)
                # Generate global prototypes for all classes (via the TGP model)
                proto_gen = TGP(list(range(args.class_num)))
                
                # Euclidean distance formula: sqrt((x1-y1)² + ... + (xn-yn)²) = sqrt(x² - 2xy + y²)
                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                # Generate one-hot encoding of labels (used to locate the corresponding class of the current sample)
                one_hot = F.one_hot(y, args.class_num).to(device)
                # Calculate margin (take the smaller of the maximum class distance and the preset threshold)
                # Calculate max_gap
                margin_threthold = 100.0
                margin = min(max_gap.item(), margin_threthold)
                dist = dist + one_hot * margin
                loss = nn.CrossEntropyLoss()(-dist, y)
                TGP_optimizer.zero_grad()
                loss.backward()
                TGP_optimizer.step()

        if args.cal_time:
            train_tgp_end_time = time.time()
            print(f"Time taken for training TGP in epoch {epoch}: {train_tgp_end_time - train_tgp_start_time:.2f} seconds")
            total_time = (train_end_time - train_start_time) + (agg_end_time - agg_start_time) + (train_tgp_end_time - train_tgp_start_time)
            print(f"Total time for local training, aggregation and TGP training in epoch {epoch}: {total_time:.2f} seconds")    

        # -------- Prepare Data for Plotting --------
        if args.draw_proto and epoch % 10 == 0:
            all_local_protos = []
            local_labels = []

            for client_id, proto_dict in local_proto_dict.items():
                for label, proto in proto_dict.items():
                    all_local_protos.append(proto.cpu().numpy())
                    local_labels.append(label)

            global_protos = []
            global_labels = []

            for label, proto in agg_global_proto.items():
                global_protos.append(proto.cpu().numpy())
                global_labels.append(label)

            # Dimensionality reduction after merging
            all_embed = np.vstack([all_local_protos, global_protos])
            tsne = TSNE(n_components=2, random_state=42, init='pca')
            embed_2d = tsne.fit_transform(all_embed)

            local_2d = embed_2d[:len(all_local_protos)]
            global_2d = embed_2d[len(all_local_protos):]

            # -------- Plotting --------
            plt.figure(figsize=(10, 8))
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
                    'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:lime', 'tab:teal', 
                    'tab:yellow', 'tab:violet', 'tab:indigo', 'tab:rose']

            for i, label in enumerate(set(local_labels)):
                color = colors[label % len(colors)]

                # local protos
                local_points = np.array([local_2d[j] for j in range(len(local_labels)) if local_labels[j] == label])
                plt.scatter(local_points[:, 0], local_points[:, 1],
                            c=color, marker='o', s=100, edgecolors='k', label=f'Local Proto - Class {label}')

                # global proto
                global_point = global_2d[global_labels.index(label)]
                plt.scatter(global_point[0], global_point[1],
                            c=color, marker='*', s=200, edgecolors='k', label=f'Global Proto - Class {label}')

            title = f"Local vs Global Class Prototypes in Epoch {epoch}"
            plt.title(title)
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            file_name = f'plot_figures/local_vs_global_prototypes_epoch_{epoch}.png'
            plt.savefig(file_name, dpi=300)
            # plt.show()

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
     
    
    acc_end_list = []
    f1_score_end_list = []
    client_accs_list = []
    for c in range(args.num_workers):
        client_acc = local_model_list[c].test(client_data[c].x.to(device), client_data[c].edge_index.to(device), client_data[c].edge_weight.to(device), client_data[c].y.to(device), client_idx_test[c].to(device))
        acc_end_list.append(client_acc)
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

    acc_global_end = sum(acc_end_list)/len(acc_end_list)
    f1_score_end = sum(f1_score_end_list)/len(f1_score_end_list)

    print("Performance of all clients on clean test set: {:.4f}".format(acc_global_end))
    round_reach_target_acc = args.epochs if round_reach_target_acc == 0 else round_reach_target_acc
    return acc_global_end, round_reach_target_acc, max_accuracy, f1_score_end

if __name__ == '__main__':
    main()