# The code is from GraphSHA

from tkinter.tix import Select

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch

@torch.no_grad()
def sampling_idx_individual_dst(class_num_list, idx_info):
    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    new_class_num_list = torch.Tensor(class_num_list)
    # Select a specified number of nodes as source nodes for each class
    sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
                        for cls_idx, samp_num in zip(idx_info, sampling_list)]
    sampling_src_idx = torch.cat(sampling_src_idx)

    # Generate corresponding destination nodes
    prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
    sampling_dst_idx = temp_idx_info[dst_idx]

    # Sorting src idx with corresponding dst idx
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]

    return sampling_src_idx, sampling_dst_idx

def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x

@torch.no_grad()
def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
    device = edge_index.device

    # Assign node index for augmented nodes
    row, col = edge_index[0], edge_index[1] 
    row, sort_idx = torch.sort(row)
    col = col[sort_idx] 
    degree = scatter_add(torch.ones_like(row), row, dim_size=total_node)     # Calculate the number of neighbors for each node
    new_row =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])   # Generate indices for new nodes
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)

    # Duplicate the edges of source nodes
    node_mask = torch.zeros(total_node, dtype=torch.bool)
    unique_src = torch.unique(sampling_src_idx)
    node_mask[unique_src] = True 
    row_mask = node_mask[row] 
    edge_mask = col[row_mask] 
    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    if len(temp[temp!=0]) != edge_dense.shape[0]:
        cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
        cut_temp = temp[temp!=0][:-cut_num]
    else:
        cut_temp = temp[temp!=0]
    edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense!= -1]
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

# Build connection edges for synthesized nodes
@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    
    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree degree distribution calculation
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col) # Calculate the degree of each node, grouped by the indices specified by col
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1) # New nodes share the same degree sampling distribution probability
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # Sample the degree for new nodes

    max_degree = degree.max().item() + 1    # Calculate maximum degree
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])    # Constrain degree upper limit

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]    # Filter valid neighbors
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)  # Generate indices for new nodes
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1) # Merge new and old edges

    return new_edge_index

@torch.no_grad()
def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False, no_mask=False, same_class_target=False, pseudo_graph=None):
    # Sample more instances from minority classes to balance the distribution
    # Determine the number of sampled nodes based on the deficit of nodes in each class
    max_num, n_cls = max(class_num_list), len(class_num_list) 
    if not max_flag: # mean
        max_num = sum(class_num_list) / n_cls
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)

    # Predictive probability normalization
    prev_out_local = F.softmax(prev_out_local/tau, dim=1)
    prev_out_local = prev_out_local.cpu() 
    
    src_idx_all = []
    dst_idx_all = []
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0 or class_num_list[cls_idx] <= 0: 
            continue

        # first sampling Select source nodes (hard samples)
        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze() # Select nodes with low predicted probabilities for their true class (i.e., hard-to-classify samples)
        
        # Ensure prob is 1D or 2D
        if prob.dim() == 0:
            prob = prob.unsqueeze(0)
        elif prob.dim() == 1:
            pass
        elif prob.dim() >= 3:
            prob = prob.view(-1, prob.shape[-1])

        src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True) 
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]] 

        # second sampling, select target classes
        if same_class_target:
            # Force the target class to be the same as the source node
            neighbor_cls = [cls_idx] * len(src_idx)
        else:
            # Original logic: select target classes based on probabilities
            conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]] # Source node's prediction distribution
            if not no_mask:
                conf_src[:,cls_idx] = 0
            neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze()
            neighbor_cls = [neighbor_cls.item()] if neighbor_cls.dim() == 0 else neighbor_cls.tolist()

        # third sampling, select target nodes
        # Select nodes in the target class that are predicted with high probabilities as the source class by the model
        neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls] 
        dst_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0] 
            dst_idx_ = train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]]
            dst_idx.append(dst_idx_)
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)

        src_idx_all.append(src_idx)
        dst_idx_all.append(dst_idx)
    
    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)
    
    return src_idx_all, dst_idx_all

