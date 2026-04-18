import sys 
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def get_total_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            iterator = iter(obj)
        except TypeError:
            pass
        else:
            size += sum([get_total_size(i, seen) for i in iterator if i is not None])
    return size / (1024 * 1024)

def get_split(args, data, device, client_idx):
    rs = np.random.RandomState(args.seed)
    perm = rs.permutation(data.num_nodes)  

    train_number = int(args.ratio_training*len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number]))   #tensor([  0,   1,   2,   3,   4]
    #initialize train_mask
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[idx_train] = True
    origin_train_index = client_idx[idx_train].tolist()     #[1024, 1721, 2099, 2009, 2354]
    
    test_number = int(args.ratio_testing*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number:train_number+test_number]))
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[idx_test] = True
    origin_test_index = client_idx[idx_test].tolist()
    
    return data, idx_train, idx_test, origin_train_index, origin_test_index

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def agg_local_proto_func(local_proto_label):
    #input: local_proto_label 
    #output: agg_proto
    agg_local_label_dict = {}
    for label, proto_list in local_proto_label.items():
        if len(proto_list) > 1:
            proto = torch.zeros_like(proto_list[0])
            for i in proto_list:
                proto += i
            agg_local_label_dict[label] = proto.detach() / len(proto_list)
        else:
            agg_local_label_dict[label] = proto_list[0].detach()
    return agg_local_label_dict

def agg_global_proto_func(local_proto_dict):
    #input: local_proto_list
    #output: global_proto
    agg_global_proto = {}   #key:label #[local_proto_1, local_proto_2]
    for key, local_proto in local_proto_dict.items():
        for label, proto in local_proto.items():
            if label not in agg_global_proto.keys():
                agg_global_proto[label] = [proto]
            else:
                agg_global_proto[label].append(proto)

    for label, proto_list in agg_global_proto.items():
        if len(proto_list) > 1:
            protos = torch.zeros_like(proto_list[0])
            for i in proto_list:
                protos += i
            agg_global_proto[label] = protos / len(proto_list)
        else:
            agg_global_proto[label] = proto_list[0]
    return agg_global_proto

def avg_per_class_acc(client_accs_list, n_class):
    '''Calculate the average precision of N models across M classes
    Input: client_accs_list
    Output: avg_acc, major_avg_acc, minor_avg_acc
    '''
    all_classes = list(range(n_class))
    all_cls_avg = {}
    for cls in all_classes:
        cls_list = []
        for client_dict in client_accs_list:
            acc = client_dict.get(cls, 0.0)
            cls_list.append(acc)
        all_cls_avg[cls] = sum(cls_list) / len(cls_list)
    return all_cls_avg


from numpy import dot
from numpy.linalg import norm
import random
from torch_geometric.data import Data

def cosine_similarity(feature1, feature2):
    return dot(feature1, feature2) / (norm(feature1) * norm(feature2))

def add_edge_dataset(client_data, num_random_edge=50, similarity_threshold=0.5, max_attempts=5000):
    """
    Enhance graph data by adding new edges based on cosine similarity.
    
    Args:
        client_data: Input graph data (PyG Data object)
        num_random_edge: Number of new edges to add (undirected)
        similarity_threshold: Minimum cosine similarity for new edges
        max_attempts: Maximum attempts to find valid edges
    
    Returns:
        Enhanced graph data with new edges
    """
    num_nodes = client_data.num_nodes
    new_edges = [[], []]
    
    # Convert existing edges to set for O(1) lookups
    existing_edges = set(map(tuple, client_data.edge_index.t().tolist()))
    
    # For tracking attempts to prevent infinite loops
    attempts = 0
    
    while len(new_edges[0]) < num_random_edge and attempts < max_attempts:
        u = random.randint(0, num_nodes-1)
        v = random.randint(0, num_nodes-1)
        
        # Skip if same node or edge already exists
        if u == v or (u, v) in existing_edges:
            attempts += 1
            continue
            
        # Calculate similarity
        sim = cosine_similarity(client_data.x[u].numpy(), client_data.x[v].numpy())
        
        if sim > similarity_threshold:
            # Add edge in both directions for undirected graph
            new_edges[0].extend([u, v])
            new_edges[1].extend([v, u])
            existing_edges.update([(u, v), (v, u)])
            
        attempts += 1
    
    if len(new_edges[0]) < num_random_edge:
        print(f"Warning: Only added {len(new_edges[0])//2} edges out of {num_random_edge} requested")
    
    # Convert to tensor and merge with original edges
    new_edge_index = torch.tensor(new_edges, dtype=torch.long)
    combined_edges = torch.cat([client_data.edge_index, new_edge_index], dim=1)
    
    # Create new data object
    restored_data = Data(
        x=client_data.x,
        edge_index=combined_edges,
        edge_attr=client_data.edge_attr if hasattr(client_data, 'edge_attr') else None,
        y=client_data.y,
        train_mask=client_data.train_mask,
        val_mask=client_data.val_mask,
        test_mask=client_data.test_mask
    )
    
    return restored_data


# import torch
# from torch_geometric.data import Data
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

def add_edge_dataset_v2(client_data, num_random_edge=30):
    """
    Enhance graph data by adding new edges based on cosine similarity.
    Selects the top num_random_edge non-existing edges with highest similarity.
    
    Args:
        client_data: Input graph data (PyG Data object)
        num_random_edge: Number of new edges to add (undirected)
    
    Returns:
        Enhanced graph data with new edges
    """
    num_nodes = client_data.num_nodes
    
    # Convert existing edges to set for O(1) lookups
    existing_edges = set(map(tuple, client_data.edge_index.t().tolist()))
    
    # Calculate pairwise cosine similarity matrix
    sim_matrix = cosine_similarity(client_data.x.numpy())
    
    # Create a list of all possible candidate edges (u, v) where u < v to avoid duplicates
    candidates = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                candidates.append((u, v, sim_matrix[u, v]))
    
    # Sort candidates by similarity in descending order
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Select top num_random_edge edges
    selected_edges = candidates[:num_random_edge]
    
    # Prepare new edges (add both directions for undirected graph)
    new_edges = [[], []]
    for u, v, _ in selected_edges:
        new_edges[0].extend([u, v])
        new_edges[1].extend([v, u])
    
    # Convert to tensor and merge with original edges
    new_edge_index = torch.tensor(new_edges, dtype=torch.long)
    combined_edges = torch.cat([client_data.edge_index, new_edge_index], dim=1)
    
    # Create new data object
    restored_data = Data(
        x=client_data.x,
        edge_index=combined_edges,
        edge_attr=client_data.edge_attr if hasattr(client_data, 'edge_attr') else None,
        y=client_data.y,
        train_mask=client_data.train_mask,
        val_mask=client_data.val_mask,
        test_mask=client_data.test_mask
    )
    
    return restored_data

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

class_colors = [
    "peru",         # 2 brownish orange
    "tomato",       # 1 red
    '#FFB347',      # 3 light orange
    "#74D174",      # 4 light green
    "#4F9ACF",      # 6 light blue
    "#9C7FB3",      # 5 light purple
    "dimgray",      # 7 gray
    "#A8D8B9",      # 8 mint green (new)
    "#DABFBA",      # 9 dusty pink (new)
    "#B0C4DE",      # 10 misty blue (new)
    "#CDAF7E",      # 11 khaki (new)
    "#87A88C",      # 12 olive green (new)
    "#A99EAA",      # 13 light purplish gray (new)
]

def visualize_node_embeddings(
    local_model, 
    client_data, 
    client_idx_train, 
    client_idx_test, 
    pseudo_graph, 
    device, 
    class_num, 
    client_id, 
    epoch, 
    output_dir="plot_figures",
    figsize=(10, 8),
    perplexity=20,
    learning_rate=50,
    n_iter=2000
):
    """s
    Visualize embeddings of local nodes and pseudo nodes (via dimensionality reduction)

    Parameters:
        local_model: Local model
        client_data: Client data (contains x, edge_index, edge_weight, y)
        client_idx_train: Training set indices
        client_idx_test: Test set indices
        pseudo_graph: Pseudo-graph data (contains x, edge_index, edge_weight)
        device: Device (e.g., 'cuda' or 'cpu')
        class_num: Number of classes (for color mapping range)
        client_id: Client ID (for title and filename)
        epoch: Current epoch (for filename)
        output_dir: Output directory (default: "plot_figures")
        figsize: Figure size (default: (10, 8))
        perplexity: Perplexity parameter for TSNE (default: 20)
        learning_rate: Learning rate for TSNE (default: 50)
        n_iter: Number of iterations for TSNE (default: 2000)
    """
    # -------- Global Settings: Set Font to Times New Roman --------
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rcParams['font.family'] = font_name
    mpl.rcParams['font.serif'] = [font_name]
    myfontsize = 19
    mpl.rcParams['font.size'] = myfontsize
    mpl.rcParams['axes.titlesize'] = myfontsize
    mpl.rcParams['axes.labelsize'] = myfontsize
    mpl.rcParams['xtick.labelsize'] = myfontsize
    mpl.rcParams['ytick.labelsize'] = myfontsize
    mpl.rcParams['legend.fontsize'] = myfontsize
    mpl.rcParams['figure.titlesize'] = myfontsize
    custom_cmap = ListedColormap(class_colors[:class_num])

    # -------- 1. Obtain All Local Node Embeddings and Predictions --------
    local_model.eval()
    with torch.no_grad():
        output_all, embed_all, _ = local_model(
            client_data.x.to(device),
            client_data.edge_index.to(device),
            client_data.edge_weight.to(device)
        )
    _, preds_all = torch.max(output_all, dim=1)
    preds_all = preds_all.cpu().numpy()
    embed_all_np = embed_all.cpu().numpy()

    # Record node information
    test_idx = client_idx_test
    train_idx = client_idx_train
    test_labels = client_data.y[test_idx].cpu().numpy()
    train_labels = client_data.y[train_idx].cpu().numpy()

    # -------- 2. Obtain Pseudo-graph Embeddings and Predictions --------
    with torch.no_grad():
        pseudo_output, pseudo_embed, _ = local_model(
            pseudo_graph.x.to(device),
            pseudo_graph.edge_index.to(device),
            pseudo_graph.edge_weight.to(device)
        )
    _, pseudo_preds = torch.max(pseudo_output, dim=1)
    pseudo_preds = pseudo_preds.cpu().numpy()
    pseudo_embed_np = pseudo_embed.cpu().numpy()
    pseudo_labels = pseudo_graph.y.cpu().numpy()

    # -------- 3. Merge all local nodes + pseudo nodes, dimensionality reduction --------
    combined_embed = np.vstack([embed_all_np, pseudo_embed_np])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init='pca',
        random_state=42
    )
    combined_2d = tsne.fit_transform(combined_embed)

    # Restore the 2D spatial positions of local, test, and pseudo embeddings separately
    local_2d = combined_2d[:len(embed_all_np)]
    pseudo_2d = combined_2d[len(embed_all_np):]
    test_2d = local_2d[test_idx]
    train_2d = local_2d[train_idx]

    # -------- 4. Plotting --------
    plt.figure(figsize=figsize)
    # Combine all points and calculate the range
    all_x = np.concatenate([local_2d[:,0], pseudo_2d[:,0]])
    all_y = np.concatenate([local_2d[:,1], pseudo_2d[:,1]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    vmin, vmax = 0, class_num - 1  # Color mapping range

    # Plot training nodes
    scatter_train = plt.scatter(
        train_2d[:, 0], train_2d[:, 1],
        c=train_labels,
        cmap=custom_cmap,
        marker='o',
        s=160,
        vmin=vmin,
        vmax=vmax,
        label='Local Train Nodes'
    )

    # Plot test nodes
    scatter_test = plt.scatter(
        test_2d[:, 0], test_2d[:, 1],
        c=test_labels,
        cmap=custom_cmap,
        marker='o',
        edgecolors='k',
        s=160,
        vmin=vmin,
        vmax=vmax,
        label='Local Test Nodes'
    )

    # Plot pseudo-graph nodes
    scatter_pseudo = plt.scatter(
        pseudo_2d[:, 0], pseudo_2d[:, 1],
        c=pseudo_labels,
        cmap=custom_cmap,
        marker='x',
        s=180,
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        label='Pseudo Graph Nodes'
    )

    cbar = plt.colorbar(label="Class Label")
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label="Class Label", fontsize=myfontsize)
    plt.title(f"Node Embeddings - Client {client_id}, Epoch {epoch}")

    # -------- Core Modification: Optimize Legend (Remove Border + Clear Train/Test Distinction) --------
    leg = plt.legend(frameon=False)
    labels = [t.get_text() for t in leg.get_texts()]
    handles = getattr(leg, "legend_handles", None)
    if handles is None:
        handles = getattr(leg, "legendHandles", None)
    
    # 2. Manually create a new legend (completely independent of original plot colors)
    from matplotlib.lines import Line2D

    custom_legend = [
        Line2D([], [], linestyle='None', marker='o', color='w', markerfacecolor='#4A90E2', markeredgecolor='none', markersize=12, label='Local Train Nodes'),
        Line2D([], [], linestyle='None', marker='o', color='w', markerfacecolor='#4A90E2', markeredgecolor='black', markeredgewidth=2, markersize=12, label='Local Test Nodes'),
        Line2D([], [], linestyle='None', marker='x', color='#4A90E2', markeredgewidth=2, markersize=12, label='Pseudo Graph Nodes')
    ]

    # 3. Draw new legend (borderless, clear, correct colors)
    plt.legend(handles=custom_legend, frameon=False, fontsize=myfontsize)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{epoch}_client_{client_id}.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()


def visualize_node_embeddings_new_data(
    local_model, 
    new_x,
    new_edge_index,
    new_edge_weight,
    _new_train_idx,
    new_y,
    client_idx_train, 
    client_idx_test, 
    pseudo_graph, 
    device, 
    class_num, 
    client_id, 
    epoch, 
    output_dir="plot_figures",
    figsize=(10, 8),
    perplexity=20,
    learning_rate=50,
    n_iter=2000
):
    """
    Visualize embeddings of local nodes and pseudo nodes (via TSNE dimensionality reduction)

    Parameters:
        local_model: Local model
        client_data: Client data (contains x, edge_index, edge_weight, y)
        client_idx_train: Training set indices
        client_idx_test: Test set indices
        pseudo_graph: Pseudo-graph data (contains x, edge_index, edge_weight)
        device: Device (e.g., 'cuda' or 'cpu')
        class_num: Number of classes (for color mapping range)
        client_id: Client ID (for title and filename)
        epoch: Current epoch (for filename)
        output_dir: Output directory (default: "plot_figures")
        figsize: Figure size (default: (10, 8))
        perplexity: Perplexity parameter for TSNE (default: 20)
        learning_rate: Learning rate for TSNE (default: 50)
        n_iter: Number of iterations for TSNE (default: 2000)
    """
    # -------- Global Settings: Set Font to Times New Roman --------
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rcParams['font.family'] = font_name
    mpl.rcParams['font.serif'] = [font_name]
    myfontsize = 19
    mpl.rcParams['font.size'] = myfontsize
    mpl.rcParams['axes.titlesize'] = myfontsize
    mpl.rcParams['axes.labelsize'] = myfontsize
    mpl.rcParams['xtick.labelsize'] = myfontsize
    mpl.rcParams['ytick.labelsize'] = myfontsize
    mpl.rcParams['legend.fontsize'] = myfontsize
    mpl.rcParams['figure.titlesize'] = myfontsize
    custom_cmap = ListedColormap(class_colors[:class_num])

    # -------- 1. Get All Local Node Embeddings and Predictions --------
    local_model.eval()
    with torch.no_grad():
        output_all, embed_all, _ = local_model(
            new_x.to(device),
            new_edge_index.to(device),
            new_edge_weight.to(device)
        )
    _, preds_all = torch.max(output_all, dim=1)
    preds_all = preds_all.cpu().numpy()
    embed_all_np = embed_all.cpu().numpy()

    # Record node information
    test_idx = client_idx_test
    train_idx = client_idx_train
    new_train_idx = _new_train_idx
    test_labels = new_y[test_idx].cpu().numpy()
    train_labels = new_y[train_idx].cpu().numpy()
    new_train_labels = new_y[new_train_idx].cpu().numpy() # Used to record the indices of new training nodes

    # -------- 2. Get Pseudo-graph Embeddings and Predictions --------
    with torch.no_grad():
        pseudo_output, pseudo_embed, _ = local_model(
            pseudo_graph.x.to(device),
            pseudo_graph.edge_index.to(device),
            pseudo_graph.edge_weight.to(device)
        )
    _, pseudo_preds = torch.max(pseudo_output, dim=1)
    pseudo_preds = pseudo_preds.cpu().numpy()
    pseudo_embed_np = pseudo_embed.cpu().numpy()
    pseudo_labels = pseudo_graph.y.cpu().numpy()

    # -------- 3. Merge all local nodes + pseudo nodes, dimensionality reduction --------
    combined_embed = np.vstack([embed_all_np, pseudo_embed_np])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init='pca',
        random_state=42
    )
    combined_2d = tsne.fit_transform(combined_embed)

    # Restore the 2D spatial positions of local, test, and pseudo embeddings separately
    local_2d = combined_2d[:len(embed_all_np)]
    pseudo_2d = combined_2d[len(embed_all_np):]
    test_2d = local_2d[test_idx]
    train_2d = local_2d[train_idx]
    new_train_2d = local_2d[new_train_idx]

    # -------- 4. Plotting --------
    plt.figure(figsize=figsize)
    all_x = np.concatenate([local_2d[:,0], pseudo_2d[:,0]])
    all_y = np.concatenate([local_2d[:,1], pseudo_2d[:,1]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    vmin, vmax = 0, class_num - 1  # Color mapping range

    # Plot training nodes
    scatter_train = plt.scatter(
        train_2d[:, 0], train_2d[:, 1],
        c=train_labels,
        cmap=custom_cmap,
        marker='o',
        s=160,
        vmin=vmin,
        vmax=vmax,
        label='Local Train Nodes'
    )

    # Plot test nodes
    scatter_test = plt.scatter(
        test_2d[:, 0], test_2d[:, 1],
        c=test_labels,
        cmap=custom_cmap,
        marker='o',
        edgecolors='k',
        s=160,
        vmin=vmin,
        vmax=vmax,
        label='Local Test Nodes'
    )

    # Plot pseudo nodes
    scatter_pseudo = plt.scatter(
        pseudo_2d[:, 0], pseudo_2d[:, 1],
        c=pseudo_labels,
        cmap=custom_cmap,
        marker='x',
        edgecolors='k',
        s=150,
        linewidths=2,
        vmin=vmin,
        vmax=vmax,
        label='Pseudo Graph Nodes'
    )

    scatter_new_pseudo = plt.scatter(
        new_train_2d[:, 0], new_train_2d[:, 1],
        c=new_train_labels,
        cmap=custom_cmap,
        marker='*',
        edgecolors='k',
        s=180,
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        # label='Pseudo Graph Nodes'
    )


    cbar = plt.colorbar(label="Class Label")
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label="Class Label", fontsize=myfontsize)
    plt.title(f"Node Embeddings - Client {client_id}, Epoch {epoch}")


    from matplotlib.lines import Line2D

    # If you have star markers (New Train Nodes), replace the above custom_legend with this:
    custom_legend = [
        Line2D([], [], linestyle='None', marker='o', color='w', markerfacecolor='#4A90E2', markeredgecolor='none', markersize=12, label='Local Train Nodes'),
        Line2D([], [], linestyle='None', marker='o', color='w', markerfacecolor='#4A90E2', markeredgecolor='black', markeredgewidth=2, markersize=12, label='Local Test Nodes'),
        Line2D([], [], linestyle='None', marker='x', color='#4A90E2', markeredgewidth=2, markersize=12, label='Pseudo Graph Nodes'),
        Line2D([], [], linestyle='None', marker='*', color='#4A90E2', markeredgecolor='black', markeredgewidth=1.5, markersize=15, label='Synthetic Nodes'),
    ]
   
    # 3. Draw new legend (borderless, clear, correct colors)
    plt.legend(handles=custom_legend, frameon=False, fontsize=myfontsize)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{epoch}_client_{client_id}.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()


class ContrastiveDiversityLoss(nn.Module):
    def __init__(self, temperature=0.1, metric='cosine'):
        super().__init__()
        self.temperature = temperature
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_similarity(self, tensor1, tensor2):
        """Calculate similarity between samples (supports cosine or dot product)"""
        if self.metric == 'cosine':
            return self.cosine(tensor1.unsqueeze(1), tensor2.unsqueeze(0))
        elif self.metric == 'dot':
            return torch.matmul(tensor1, tensor2.T)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def feature_augmentation(self, features, p=0.1):
        """
        Args:
            features: Node feature matrix [N, D]
            p: Mask/noise ratio
        """
        mask = torch.rand(features.size()) < p
        noise = torch.randn_like(features) * 0.1
        features_aug = features.clone()
        features_aug[mask] += noise[mask]
        return features_aug

    def forward(self, features, noises=None):
        """
        Args:
            features: Input feature tensor [N, D] (N: number of samples, D: feature dimension)
            noises: Noise input (optional, used for weighting if provided)
        Returns:
            Contrastive diversity loss value
        """
        N = features.size(0)
        
        # 1. Data augmentation to generate positive sample pairs (simplified here: directly copy original samples)
        # (In practical applications, replace with real data augmentation, e.g., random cropping, rotation, etc.)
        features_aug = self.feature_augmentation(features)
        
        # 2. Compute similarity matrix
        sim = self.compute_similarity(features, features_aug)  # [N, N]
        sim /= self.temperature  # Temperature scaling
        
        # 3. Build contrastive learning target
        # Diagonal elements are positive sample pairs, others are negative samples
        pos_mask = torch.eye(N, dtype=torch.bool, device=features.device)  # Diagonal is True
        neg_mask = ~pos_mask
        
        # 4. Compute contrastive loss (InfoNCE)
        exp_sim = torch.exp(sim)
        # pos_sim = exp_sim[pos_mask].sum()  # Numerator: similarity of positive samples
        # neg_sim = exp_sim[neg_mask].sum()  # Denominator: similarity of all negative samples
        # loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        

        # Compute loss for each sample, then take the average
        pos_sim_per_sample = exp_sim[pos_mask]  # [N]
        neg_sim_per_sample = exp_sim[neg_mask].view(N, N-1).sum(dim=1)  # [N]
        loss_per_sample = -torch.log(pos_sim_per_sample / (pos_sim_per_sample + neg_sim_per_sample))
        loss = loss_per_sample.mean()  # Take average
                
        # 5. Optional: Weight by noise distance (keep original logic)
        if noises is not None:
            noise_dist = torch.cdist(noises, noises, p=2)  # L2 distance matrix [N, N]
            max_dist = noise_dist.max()
            min_dist = noise_dist.min()
            # Normalize to [0,1], the larger the noise difference, the closer the weight is to 1
            weight = (noise_dist.mean() - min_dist) / (max_dist - min_dist + 1e-8)
            weighted_loss = loss * weight
            return weighted_loss
        
        return loss


if __name__ == "__main__":
    # Initialization
    diversity_loss = ContrastiveDiversityLoss(temperature=0.1, metric='cosine')

    # Input features (assuming batch_size=32, feature_dim=128)
    features = torch.randn(32, 128)  
    noises = torch.randn(32, 64)

    # Compute loss
    loss = diversity_loss(features, None)
    print(loss)