import networkx as nx
from src.topologies.star import StarTopology
from src.topologies.hierarchical import HierarchicalTopology

def check_star_invariant(topology: StarTopology, num_clients: int):
    assert topology.num_clients == num_clients
    assert topology.graph is not None
    assert nx.is_connected(topology.graph), "Star topology must be connected"
    assert len(topology.get_server_connected_clients()) == num_clients

def check_hierarchical_invariant(topology: HierarchicalTopology, num_clients: int):
    assert topology.num_clients == num_clients
    assert len(topology.get_server_connected_clients()) == topology.num_clusters, "Server should connect to exactly num_clusters heads"
    # Every client should map to exactly 1 cluster head
    for i in range(num_clients):
        heads = topology.get_neighbors(i)
        assert len(heads) == 1, "Each client must have exactly 1 cluster head"
        assert heads[0] < 0, "Cluster head ID must be negative"
