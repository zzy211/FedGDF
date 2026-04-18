from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import community as community_louvain
import random
from tqdm import tqdm
import pymetis as metis
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import pandas as pd


import torch_geometric
def split_communities(data, clients):
    G = to_networkx(data, to_undirected=True, node_attrs=['x', 'y'])
    communities = sorted(nx.community.asyn_fluidc(G, clients, max_iter=5000, seed=12345))

    node_groups = []
    for com in communities:
        node_groups.append(list(com))
    list_of_clients = []

    for i in range(clients):
        list_of_clients.append(from_networkx(G.subgraph(node_groups[i]).copy()))

    return list_of_clients

def split_Metis(args,data):
    """
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.drop_edge = 0
    ovlap = args.overlapping_rate
    drop_edge = args.drop_edge
    client_num = args.num_workers

    sampling_rate = (np.ones(client_num) -
                          ovlap) / client_num

    data.index_orig = torch.arange(data.num_nodes)


    print("Graph to Networkx")
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    print("Calculating  partition")
    client_node_idx = {idx: [] for idx in range(client_num)}

    n_cuts, membership = metis.part_graph(G, client_num)    #Partition the graph using metis and assign nodes to different clients
    indices = []
    for i in range(client_num):
        client_indices = np.where(np.array(membership) == i)[0]
        indices.append(client_indices)
    indices = np.concatenate(indices)



    sum_rate = 0
    for idx, rate in enumerate(sampling_rate):
        client_node_idx[idx] = indices[round(sum_rate *
                                             data.num_nodes):round(
            (sum_rate + rate) *
            data.num_nodes)]
        sum_rate += rate

    if ovlap:
        ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
        for idx in client_node_idx:
            client_node_idx[idx] = np.concatenate(
                (client_node_idx[idx], ovlap_nodes))

    # Drop_edge index for each client
    if drop_edge:
        ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
        ovlap_edge_ind = np.random.permutation(
            ovlap_graph.number_of_edges())
        drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                         drop_edge)]
        drop_client = [
            drop_all[s:s + round(len(drop_all) / client_num)]
            for s in range(0, len(drop_all),
                           round(len(drop_all) / client_num))
        ]

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        if drop_edge:
            sub_g.remove_edges_from(
                np.array(ovlap_graph.edges)[drop_client[owner]])
        graphs.append(from_networkx(sub_g))

    return graphs



def split_Random(args,data):
    """
    Count the total number of nodes in train_node during node partitioning, then randomly assign them to clients. After partitioning, assign the corresponding subgraphs (from_networkx) of nodes to each client.
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.drop_edge = 0
    ovlap = args.overlapping_rate
    drop_edge = args.drop_edge
    client_num = args.num_workers

    #calculate sample rate for each client
    sampling_rate = (np.ones(client_num) -
                          ovlap) / client_num

    #add index to data sample
    data.index_orig = torch.arange(data.num_nodes)

    #transform data into networkx graph structure
    print("Graph to Networkx")
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    
    #set node attributes
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    
    print("Calculating  partition")
    #initial node index of each client
    client_node_idx = {idx: [] for idx in range(client_num)}
    indices = np.random.permutation(data.num_nodes)
    #partition according to sampling rate
    sum_rate = 0
    for idx, rate in enumerate(sampling_rate):
        client_node_idx[idx] = indices[round(sum_rate *
                                             data.num_nodes):round(
            (sum_rate + rate) *
            data.num_nodes)]
        sum_rate += rate

    #if ovlap, partition overlapping node
    if ovlap:
        ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
        for idx in client_node_idx:
            client_node_idx[idx] = np.concatenate(
                (client_node_idx[idx], ovlap_nodes))

    # Drop_edge index for each client
    if drop_edge:
        ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
        ovlap_edge_ind = np.random.permutation(
            ovlap_graph.number_of_edges())
        drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                         drop_edge)]
        drop_client = [
            drop_all[s:s + round(len(drop_all) / client_num)]
            for s in range(0, len(drop_all),
                           round(len(drop_all) / client_num))
        ]

    #create local subgraph of each client
    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        if drop_edge:
            sub_g.remove_edges_from(
                np.array(ovlap_graph.edges)[drop_client[owner]])
        graphs.append(from_networkx(sub_g))

    return graphs, client_node_idx



def split_graph_kernal(args, data):
    args.drop_edge = 0
    ovlap = args.overlapping_rate
    drop_edge = args.drop_edge
    client_num = args.num_workers
    classes_num = args.class_num
    #1.add index to data sample
    data.index_orig = torch.arange(data.num_nodes)
    #2.transform data into networkx graph structure
    print("Graph to Networkx")
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    #3.set node attributes
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    #4.calculate partition of each client
    min_size = 0
    min_limit = 10
    alpha = args.kernal_alpha 
    while min_size < min_limit:
        print("kernal alpha:", alpha)
        #4.1 consruct similarity matrix (accordding to random walk kernal)
        A = nx.adjacency_matrix(G).todense()
        D = np.diag(np.sum(A, axis=1).A1)
        D_inv = np.linalg.inv(D)
        P = D_inv @ A   #P = D^{-1}A
        lambda_factor = 0.5 #lambda define the weight of different walks
        I = np.eye(G.number_of_nodes())
        W1 = P
        W2 = P @ P
        K = I + lambda_factor * W1 + (lambda_factor ** 2) * W2

        #4.2 adjust similarity matrix according to alpha
        adjusted_similarity_matrix = alpha * K + (1-alpha) * np.mean(K)
        # Convert adjusted_similarity_matrix to np.array
        adjusted_similarity_matrix = np.asarray(adjusted_similarity_matrix)

        #4.3 cluster according to matrix, and paritition to clients
        kmeans = KMeans(n_clusters=classes_num, init='k-means++', n_init=100, max_iter=1000, random_state=42)
        labels = kmeans.fit_predict(adjusted_similarity_matrix)
        
        idx_class = [[] for _ in range(classes_num)]
        for c_id in range(classes_num):
            idx_class[c_id] = np.where(labels == c_id)[0]
            np.random.shuffle(idx_class[c_id])
        
        total_nodes = data.num_nodes
        avg_nodes_per_client = total_nodes // client_num
        all_indices = np.concatenate(idx_class, axis=0)
        idx_client = [all_indices[i * avg_nodes_per_client:(i + 1) * avg_nodes_per_client] for i in range(client_num)]
        min_size = min([len(idx_client[i]) for i in range(client_num)])
                
    
    #5 if overlap, partition overlapping rate
    if ovlap:
        # Calculate the number of overlapping nodes
        overlap_count = int(data.num_nodes * ovlap)
        indices = np.random.permutation(data.num_nodes)
        ovlap_nodes = indices[-overlap_count:]
        # Assign overlapping nodes to each client
        for idx in range(client_num):
            idx_client[idx] = np.concatenate((idx_client[idx], ovlap_nodes))
        # Ensure uniqueness
        for idx in range(client_num):
            idx_client[idx] = np.unique(idx_client[idx])
    #6 Drop_edge index for each client
    if drop_edge:
        ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
        ovlap_edge_ind = np.random.permutation(
            ovlap_graph.number_of_edges())
        drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                         drop_edge)]
        drop_client = [
            drop_all[s:s + round(len(drop_all) / client_num)]
            for s in range(0, len(drop_all),
                           round(len(drop_all) / client_num))
        ]
    #7 create local subgraph for each client
    graphs = []
    for i in range(client_num):
        nodes = idx_client[i]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        if drop_edge:
            sub_g.remove_edges_from(
                np.array(ovlap_graph.edges)[drop_client[i]])
        graphs.append(from_networkx(sub_g))
    return graphs
    


def split_dirichlet(args, data):
    args.drop_edge = 0
    ovlap = args.overlapping_rate
    drop_edge = args.drop_edge
    client_num = args.num_workers
    classes_num = args.class_num

    #1.add index to data sample
    data.index_orig = torch.arange(data.num_nodes)

    #2.transform data into networkx graph structure
    print("Graph to Networkx")
    G = to_networkx(
        data,
        # node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        node_attrs=['x', 'y'],
        to_undirected=True)
    
    #3.set node attributes
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    
    #4.calculate partition of each client
    #4.1 partition node index of each client
    print("Calculating  partition")
    min_size = 0
    min_limit = 10
    labels = data.y.numpy()
    alpha_dir = args.dirichlet_alpha
    while min_size < min_limit:
        idx_client = [[] for _ in range(client_num)]
        for k in range(classes_num):
            idxs_k = np.where(labels == k)[0]
            np.random.shuffle(idxs_k)
            partition_rate = np.random.dirichlet(np.repeat(alpha_dir, client_num))
            partition_rate = partition_rate / partition_rate.sum()
            split_point = (np.cumsum(partition_rate) * len(idxs_k)).astype(int)[:-1]
            idx_client = [j_list + k_list.tolist() for j_list, k_list in zip(idx_client, np.split(idxs_k, split_point))]
        min_size = min([len(idx_c) for idx_c in idx_client])
    
    for i in range(client_num):
            np.random.shuffle(idx_client[i])
            
    if args.same_size_dataset == 1:
        total_nodes = data.num_nodes
        avg_nodes_per_client = total_nodes // client_num
        all_indices = np.concatenate(idx_client, axis=0)
        idx_client = [all_indices[i * avg_nodes_per_client:(i + 1) * avg_nodes_per_client] for i in range(client_num)]
    
    #4.2 if overlap, partition overlapping rate
    if ovlap:
        # Calculate the number of overlapping nodes
        overlap_count = int(data.num_nodes * ovlap)
        indices = np.random.permutation(data.num_nodes)
        ovlap_nodes = indices[-overlap_count:]
        # Assign overlapping nodes to each client
        for idx in range(client_num):
            idx_client[idx] = np.concatenate((idx_client[idx], ovlap_nodes))
        # Ensure uniqueness
        for idx in range(client_num):
            idx_client[idx] = np.unique(idx_client[idx])
    #4.3 Drop_edge index for each client
    if drop_edge:
        ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
        ovlap_edge_ind = np.random.permutation(
            ovlap_graph.number_of_edges())
        drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                         drop_edge)]
        drop_client = [
            drop_all[s:s + round(len(drop_all) / client_num)]
            for s in range(0, len(drop_all),
                           round(len(drop_all) / client_num))
        ]
    #4.4 create local subgraph for each client
    graphs = []
    for i in range(client_num):
        nodes = idx_client[i]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        if drop_edge:
            sub_g.remove_edges_from(
                np.array(ovlap_graph.edges)[drop_client[i]])
        graphs.append(from_networkx(sub_g))

    #4.5 print the number of samples per class for each client
    client_class_counts = np.zeros((client_num, classes_num))
    for i in range(client_num):
        client_labels = labels[idx_client[i]]
        for j in range(classes_num):
            client_class_counts[i][j] = np.sum(client_labels == j)
    # print("client samples distribution matrix:")
    # print(client_class_counts)
    df = pd.DataFrame(client_class_counts, columns=[f'Class {c}' for c in range(classes_num)], index=[f'Client {i}' for i in range(client_num)])
    print("\nClient samples distribution matrix:")
    print(df.to_csv(index=True, header=True))

    #如果需要存储到指定的CSV文件中
    # output_path = "client_class_distribution.csv"  # 指定文件路径和名称
    # df.to_csv(output_path, index=True, header=True)  # 直接调用to_csv保存，不通过print
    return graphs, idx_client


def split_Louvain(args,data):
    """
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.delta = 40

    delta= args.delta
    client_num = args.num_workers
    data.index_orig = torch.arange(data.num_nodes)
    print("Graph to Networkx")
    # G = to_networkx(
    #     data,
    #     node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
    #     to_undirected=True)

    node_attrs = ['x', 'y', 'train_mask', 'val_mask', 'test_mask']


    G = to_networkx(data, node_attrs=node_attrs, to_undirected=True)
    #partition = community_louvain.best_partition(G)
    #Large_data_list = ['Reddit','Reddit2','Yelp','Flickr']
    Large_data_list = ['Reddit']
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in tqdm(range(nx.number_of_nodes(G)))]),
                           name="index_orig")




    # with tqdm(desc="Calculating community partition", total= total) as pbar:
    #     partition = community_louvain.best_partition(G)
    #     pbar.update(1)
    resolution_value = args.louvain_alpha
    print("Calculating community partition")
    if args.dataset in Large_data_list: #对于大数据集使用不同的resolution参数
        partition = community_louvain.best_partition(G, resolution = 0.1 * resolution_value)    #resolution越小，更多的节点将会合并到一个社区，产生较大的社区
    else:
        partition = community_louvain.best_partition(G, resolution = resolution_value) #返回一个字典，字典的key是检点，value是社区编号


    cluster2node = {}   #存储每个社区对应的节点
    for node in partition:
        cluster = partition[node]
        if cluster not in cluster2node:
            cluster2node[cluster] = [node]
        else:
            cluster2node[cluster].append(node)

    #确定每个客户端的最大节点数，并将过大的社区进行拆分
    max_len = len(G) // client_num - delta
    max_len_client = len(G) // client_num

    tmp_cluster2node = {}
    for cluster in cluster2node:
        while len(cluster2node[cluster]) > max_len:
            tmp_cluster = cluster2node[cluster][:max_len]   #从当前社区中取出前max_len个节点，形成一个新的社区
            tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                             1] = tmp_cluster #确保新的社区编号唯一
            cluster2node[cluster] = cluster2node[cluster][max_len:] #更新原社区，只保留剩余的节点。
    cluster2node.update(tmp_cluster2node)

    #将社区按大小分配给客户端，确保每个客户端获得的节点数在允许范围内
    orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
    orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

    client_node_idx = {idx: [] for idx in range(client_num)}
    idx = 0
    for (cluster, node_list) in orderedc2n:
        while len(node_list) + len(
                client_node_idx[idx]) > max_len_client + delta:
            idx = (idx + 1) % client_num
        client_node_idx[idx] += node_list
        idx = (idx + 1) % client_num

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        graphs.append(from_networkx(nx.subgraph(G, nodes)))

    return graphs















def turn_to_pyg_data(client_graphs):
    client_data = []
    for i in range(len(client_graphs)):
        client_data.append(from_networkx(client_graphs[i]))

    return client_data


def train_test_split(data, client_id, split_percentage):
    mask = torch.randn((data.num_nodes)) < split_percentage
    nmask = torch.logical_not(mask)
    train_mask = mask
    test_mask = nmask
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


def trainer(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    #print(data.x.shape)
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return loss, test_acc


def tester(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


def tester2(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    f1 = f1_score(y_true=data.y[data.test_mask], y_pred=pred[data.test_mask], average='macro', zero_division=1)
    precision = precision_score(y_true=data.y[data.test_mask], y_pred=pred[data.test_mask], average='macro',
                                zero_division=1)
    recall = recall_score(y_true=data.y[data.test_mask], y_pred=pred[data.test_mask], average='macro', zero_division=1)
    return test_acc, f1, precision, recall


class EarlyStopping:
    def __init__(self, patience=20, change=0., path='euclid_model', mode='minimize'):
        """
        patience: Waiting threshold for val loss to improve.
        change: Minimum change in the model's quality.
        path: Path for saving the model to.
        """
        self.patience = patience
        self.change = change
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.mode = mode

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.change and self.mode == "minimize":
            self.counter += 1

            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score > self.best_score + self.change and self.mode == "maximize":
            self.counter += 1

            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0
