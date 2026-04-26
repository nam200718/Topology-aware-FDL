import networkx as nx
from src.core.interfaces import Topology
from src.topologies.star import StarTopology
from src.topologies.ring import RingTopology
from src.topologies.gossip import GossipTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.topologies.layered import LayeredTopology

def check_star_invariant(topology: StarTopology, num_clients: int):
    assert topology.num_clients == num_clients
    assert topology.graph is not None
    assert nx.is_connected(topology.graph), "Star topology must be connected"
    assert len(topology.get_server_connected_clients()) == num_clients

def check_ring_invariant(topology: RingTopology, num_clients: int):
    assert topology.num_clients == num_clients
    assert topology.graph is not None
    assert nx.is_connected(topology.graph), "Ring topology must be connected"
    for i in range(num_clients):
        assert len(topology.get_neighbors(i)) == 2, "Each Ring node must have degree 2"

def check_gossip_invariant(topology: GossipTopology, num_clients: int):
    assert topology.num_clients == num_clients
    assert topology.graph is not None
    assert nx.is_connected(topology.graph), "Gossip topology must be connected"

def check_hierarchical_invariant(topology: HierarchicalTopology, num_clients: int):
    assert topology.num_clients == num_clients
    assert len(topology.get_server_connected_clients()) == topology.num_clusters, "Server should connect to exactly num_clusters heads"
    # every client should map to exactly 1 cluster head
    for i in range(num_clients):
        heads = topology.get_neighbors(i)
        assert len(heads) == 1, "Each client must have exactly 1 cluster head"
        assert heads[0] < 0, "Cluster head ID must be negative"

def check_layered_invariant(topology: LayeredTopology, num_clients: int):
    assert topology.num_clients == num_clients, "Client count mismatch"
    # Every leaf client maps to exactly one parent
    for i in range(num_clients):
        parents = topology.get_neighbors(i)
        assert len(parents) == 1, f"Client {i} must have exactly 1 parent, got {len(parents)}"
        assert parents[0] < 0, f"Parent of client {i} must be a negative ID, got {parents[0]}"
    # Every intermediate node maps to exactly one parent
    for node_id in topology.get_all_intermediate_ids():
        parents = topology.get_neighbors(node_id)
        assert len(parents) == 1, f"Intermediate node {node_id} must have exactly 1 parent"
    # Root has no parent
    assert len(topology.get_neighbors(topology.root_id)) == 0, "Root must have no parent"
    # Total leaf count equals num_clients
    assert topology.layers[0] == num_clients, "Layer 0 size must equal num_clients"
