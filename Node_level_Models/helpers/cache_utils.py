import torch
from torch_geometric.data import Data
class PseudoGraphCache:
    def __init__(self, device, max_size=5):
        self.cache = []
        self.device = device
        self.max_size = max_size
    
    def add(self, graph):
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)
        self.cache.append(graph)
    
    def get_last_graph(self):
        if len(self.cache) > 0:
            return self.cache[-1]
        else:
            return None
    
    def get_merged_graph(self):
        if not self.cache:
            return None

        merged_graph = self.cache[0].clone()
        offset = len(merged_graph.x)
        if merged_graph.edge_weight is None:
            merged_graph.edge_weight = torch.ones(merged_graph.edge_index.size(1), device=self.device)

        for graph in self.cache[1:]:
            merged_graph.x = torch.cat([merged_graph.x, graph.x], dim=0)
            new_edges = graph.edge_index + offset
            merged_graph.edge_index = torch.cat([merged_graph.edge_index, new_edges], dim=1)

            if graph.edge_weight is None:
                graph.edge_weight = torch.ones(graph.edge_index.size(1), device=self.device)
            merged_graph.edge_weight = torch.cat([merged_graph.edge_weight, graph.edge_weight])

            merged_graph.y = torch.cat([merged_graph.y, graph.y], dim=0)
            offset += len(graph.x)
        
        return merged_graph.to(self.device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create 3 test pseudo-graphs
    def create_test_graph(node_num, class_num, edge_prob=0.3):
        x = torch.randn(node_num, 16).to(device)
        y = torch.randint(0, class_num, (node_num,)).to(device)
        edge_index = (torch.rand(node_num, node_num) < edge_prob).nonzero().t().to(device)
        return Data(x=x, edge_index=edge_index, y=y)

    graph1 = create_test_graph(100, 10)
    graph2 = create_test_graph(150, 10)
    graph3 = create_test_graph(80, 10)
    cache = PseudoGraphCache(device=device, max_size=3)

    cache.add(graph1)
    cache.add(graph2)
    print(f"Current cache size: {len(cache.cache)}")

    cache.add(graph3)
    print(f"Current cache size: {len(cache.cache)}")
    cache.add(create_test_graph(200, 10))
    print(f"Current cache size: {len(cache.cache)}")

    merged_graph = cache.get_merged_graph()

    print("\nMerged Graph Information:")
    print(f"Number of Nodes: {len(merged_graph.x)}")
    print(f"Number of Edges: {merged_graph.edge_index.size(1)}")
    print(f"Label Distribution: {torch.bincount(merged_graph.y)}")

    # Verify edge index validity
    assert (merged_graph.edge_index < len(merged_graph.x)).all(), "Edge index out of bounds!"