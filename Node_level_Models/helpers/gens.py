#用的是GraphSHA里面的代码

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
    # Compute # of source nodes
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

def saliency_mixup(x, pseudo_x, sampling_src_idx, sampling_dst_idx, lam):
    device = x.device
    sampling_src_idx, sampling_dst_idx = sampling_src_idx.to(device), sampling_dst_idx.to(device)
    pseudo_x, lam = pseudo_x.to(device), lam.to(device)
    mixed_node = lam * x[sampling_src_idx] + (1 - lam) * pseudo_x[sampling_dst_idx]
    return torch.cat([x, mixed_node], dim=0)

# Merge test nodes with locally sampled data points
def test_saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :]
    new_dst = x[sampling_dst_idx.to(x.device), :]
    lam = lam.to(x.device)
    mixed_node = lam * new_src + (1-lam) * new_dst
    return mixed_node

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
def neighbor_sampling(total_node, edge_index, sampling_src_idx):
    '''
    Input:
        Total number of nodes: 3
        edge_index: [[0, 2, 1],
                      [2, 1, 0]]
        Source node: [2]
    Output:
        edge_index: [[0, 2, 1, 2],
                      [2, 1, 0, 3]]
    '''
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    new_edge_index = [[], []]
    new_node = total_node
    for src_idx in sampling_src_idx:
        new_edge_index[0].append(src_idx)
        new_edge_index[1].append(new_node)
        new_node += 1
    new_edge_index = torch.tensor(new_edge_index, device=device)
    new_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    return new_edge_index
    

# Output values
# Source node: position in the entire local graph
# Target node: position in the pseudo graph
@torch.no_grad()
def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, test_idx, tau=2, max_flag=False, no_mask=False, same_class_target=False, pseudo_graph=None):
    # Sample more instances from minority classes to balance the distribution
    # Determine the number of sampled nodes based on the node deficit of each class
    max_num, min_num, n_cls = max(class_num_list), min(class_num_list), len(class_num_list) 
    if not max_flag: # mean
        max_num = sum(class_num_list) / n_cls
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    # Ensure the minimum number of samples is 10
    # When running on Reddit, max is set to min_num
    sampling_list = torch.clamp(sampling_list, min=10, max=max(11, min_num*5))

    # Normalize predicted probabilities
    prev_out_local = F.softmax(prev_out_local/tau, dim=1)
    prev_out_local = prev_out_local.cpu() 

    src_idx_all = []
    dst_idx_all = []

    for cls_idx, num in enumerate(sampling_list):
        # print(f"cls_idx: {cls_idx}, num: {num}")
        num = int(num.item())
        if num <= 0 or class_num_list[cls_idx] <= 0: 
            continue
        # first sampling Select source nodes (hard samples)
        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze() # Select nodes with low predicted probabilities for their own class (i.e., hard-to-classify samples)
        if prob.dim() == 0:
            prob = prob.unsqueeze(0)
        src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True) 
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]] 

        # second sampling, select target classes
        if same_class_target:
            # Force the target class to be identical to the source node's class
            neighbor_cls = [cls_idx] * len(src_idx)
        else:
            # Original logic: Select target classes based on probabilities
            conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]] # Predicted distribution of the source node
            if not no_mask:
                conf_src[:,cls_idx] = 0
            neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze()
            neighbor_cls = [neighbor_cls.item()] if neighbor_cls.dim() == 0 else neighbor_cls.tolist()

        # Find target nodes from pseudo-data
        neighbor = [pseudo_graph.y == cls for cls in neighbor_cls]
        dst_idx = []
        for i, item in enumerate(neighbor):
            valid_indices = torch.nonzero(item, as_tuple=True)[0]  # Return indices that meet the conditions
            selected_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
            dst_idx.append(selected_idx)
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)

        src_idx_all.append(src_idx)
        dst_idx_all.append(dst_idx)
    
    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)
    
    return src_idx_all, dst_idx_all



@torch.no_grad()
def sampling_test_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, test_idx, tau=2, max_flag=False, no_mask=False, same_class_target=False, pseudo_graph=None):
   
    max_num, n_cls = max(class_num_list), len(class_num_list) 
    if not max_flag: # mean
        max_num = sum(class_num_list) / n_cls
    max_num = max_num + 10
    # sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    sampling_list = 10 * torch.ones(n_cls)

    # Normalize predicted probabilities
    prev_out_local = F.softmax(prev_out_local/tau, dim=1)
    prev_out_local = prev_out_local.cpu() 

    src_idx_all = []
    dst_idx_all = []
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0: 
            continue
        # first sampling Select source nodes (hard samples)
        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze() # Select nodes with low predicted probabilities for their own class (i.e., hard-to-classify samples)
        src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True) 
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]] 

        # second sampling, select target classes
        if same_class_target:
            # Force the target class to be identical to the source node's class
            neighbor_cls = [cls_idx] * len(src_idx)
        else:
            # Original logic: Select target classes based on probabilities
            conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]] # Predicted distribution of the source node
            if not no_mask:
                conf_src[:,cls_idx] = 0
            neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze()
            neighbor_cls = [neighbor_cls.item()] if neighbor_cls.dim() == 0 else neighbor_cls.tolist()

        # third sampling Select target nodes
        # Select nodes in the target class with high predicted probabilities for the source class by the model
        # neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls] 
        # Restrict target nodes to be selected from test_idx
        neighbor = [prev_out_local[test_idx, cls] for cls in neighbor_cls] # Do not enforce target classes
        dst_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0] 
            dst_idx.append(test_idx[dst_idx_local])
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)

        src_idx_all.append(src_idx)
        dst_idx_all.append(dst_idx)
    
    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)
    
    return src_idx_all, dst_idx_all