import torch
from collections import defaultdict
import itertools

def change_isolate_label(miss_neighbor_label):
    M = max(miss_neighbor_label)  # Maximum class label
    label_to_nodes = defaultdict(list)
    for node_id, label in enumerate(miss_neighbor_label):
        label_to_nodes[label].append(node_id)

    for label, nodes in list(label_to_nodes.items()):  # Use list() to copy the current key-value pairs
        if len(nodes) == 1:  # If the class has only one node
            miss_neighbor_label[nodes[0]] = M + 1  # Change the class to M+1
            label_to_nodes[M + 1].append(nodes[0])  # Add to the new class
            del label_to_nodes[label]  # Delete the original class
    return label_to_nodes

def hier_with_specified_sizel(label_to_nodes):
    min1, min2 = float('inf'), float('inf')
    min1_idx, min2_idx = -1, -1
    for label, nodes in list(label_to_nodes.items()):
        if len(nodes) < min1:  # If the current value is less than the minimum
            min2, min2_idx = min1, min1_idx
            min1, min1_idx = len(nodes), label
        elif len(nodes) < min2:  # If the current value is between the minimum and the second minimum
            min2, min2_idx = len(nodes), label
    if min1 >= 2:
        return label_to_nodes
    else:
        label_to_nodes[min1_idx].extend(label_to_nodes[min2_idx])   # Merge nodes
        del label_to_nodes[min2_idx]    # Delete the merged class
    return label_to_nodes


def after_cluster_feature(label_to_nodes, miss_neighbor_feature):
    node_to_average = {}
    # Calculate the total number of classes
    total_categories = len(label_to_nodes)
    for label, nodes in label_to_nodes.items():
        # Compute feature mean by class
        avg_feature = torch.mean(torch.stack([miss_neighbor_feature[node] for node in nodes]), dim=0)
        for node in nodes:
            node_to_average[node] = avg_feature
    return node_to_average, total_categories

def hier_to_labels(miss_neighbor_feature, tensor_miss_neighbor_label):
    miss_neighbor_label = [tensor.item() for tensor in tensor_miss_neighbor_label]
    label_to_nodes = change_isolate_label(miss_neighbor_label)

    label_to_nodes = hier_with_specified_sizel(label_to_nodes)
    node_to_average, total_categories = after_cluster_feature(label_to_nodes, miss_neighbor_feature)
    return node_to_average, total_categories



if __name__ == "__main__":
    miss_neighbor_feature = [torch.tensor([0.2, 0.1]), torch.tensor([0.4, 0.1]), torch.tensor([0.2, 0.5]),  torch.tensor([0.2, 0.5]), torch.tensor([0.2, 0.5]), torch.tensor([0.2, 0.5])]
    miss_neighbor_label = torch.tensor([0, 1, 1, 2, 0, 2])
    node_to_average, total_categories = hier_to_labels(miss_neighbor_feature, miss_neighbor_label)
    for key, value in node_to_average.items():
        print("node:",key," value:", value)
    
    


    