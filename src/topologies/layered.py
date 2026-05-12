from typing import List, Dict, Optional
import numpy as np
from src.core.interfaces import Topology


class LayeredTopology(Topology):
    """
    [DEEP HIERARCHICAL] Layered DAG Topology
    
    Architecture:
    - Multi-layer Directed Acyclic Graph (DAG) resembling a Neural Network.
    - Includes intermediate aggregation layers between clients and server.
    - Features "Intra-Layer Gossip" for lateral knowledge sharing at each level.
    """

    def __init__(self, layers: Optional[List[int]] = None, gossip_steps: int = 1):
        if layers is None:
            layers = [10, 4, 2, 1]
        self.layers = layers
        self.gossip_steps = gossip_steps
        self.num_clients = 0

        # Mapping: node_id -> parent_id (one level up)
        self.parent_of: Dict[int, int] = {}
        # Mapping: node_id -> list of child node_ids (one level down)
        self.children_of: Dict[int, List[int]] = {}
        # All node IDs organized by layer index
        self.layer_nodes: Dict[int, List[int]] = {}
        # Intra-layer peer connections (ring neighbors)
        self.layer_peers: Dict[int, List[int]] = {}
        # Reverse lookup: node_id -> layer index
        self.node_to_layer: Dict[int, int] = {}
        # The root server node ID
        self.root_id: int = -1

    def build(self, num_clients: int, seed: int) -> None:
        self.num_clients = num_clients

        if self.layers[0] != num_clients:
            raise ValueError(
                f"First layer size ({self.layers[0]}) must equal num_clients ({num_clients})"
            )
        if self.layers[-1] != 1:
            raise ValueError(
                f"Last layer size must be 1 (the global server), got {self.layers[-1]}"
            )

        rng = np.random.RandomState(seed)

        # Layer 0: real client IDs are 0 .. num_clients-1
        self.layer_nodes[0] = list(range(num_clients))
        for cid in self.layer_nodes[0]:
            self.node_to_layer[cid] = 0

        # Assign negative IDs to intermediate layers (1, 2, ..., L-1)
        next_neg_id = -2
        for layer_idx in range(1, len(self.layers)):
            count = self.layers[layer_idx]
            ids = []
            for _ in range(count):
                ids.append(next_neg_id)
                self.node_to_layer[next_neg_id] = layer_idx
                next_neg_id -= 1
            self.layer_nodes[layer_idx] = ids

        # The single node in the last layer is the root server
        self.root_id = self.layer_nodes[len(self.layers) - 1][0]

        # Initialize children_of for all non-leaf nodes
        for layer_idx in range(1, len(self.layers)):
            for node_id in self.layer_nodes[layer_idx]:
                self.children_of[node_id] = []

        # Assign parent-child relationships layer by layer using round-robin
        for layer_idx in range(len(self.layers) - 1):
            child_nodes = self.layer_nodes[layer_idx]
            parent_nodes = self.layer_nodes[layer_idx + 1]
            num_parents = len(parent_nodes)

            shuffled = list(child_nodes)
            rng.shuffle(shuffled)

            for i, child_id in enumerate(shuffled):
                parent_id = parent_nodes[i % num_parents]
                self.parent_of[child_id] = parent_id
                self.children_of[parent_id].append(child_id)

        # Build intra-layer peer connections (ring topology within each layer)
        for layer_idx in range(len(self.layers)):
            nodes = self.layer_nodes[layer_idx]
            n = len(nodes)
            for i, node_id in enumerate(nodes):
                if n <= 1:
                    # Single node: no peers
                    self.layer_peers[node_id] = []
                elif n == 2:
                    # Two nodes: each is the other's peer
                    self.layer_peers[node_id] = [nodes[1 - i]]
                else:
                    # Ring: connect to left and right neighbor
                    left = nodes[(i - 1) % n]
                    right = nodes[(i + 1) % n]
                    self.layer_peers[node_id] = [left, right]

    def get_neighbors(self, node_id: int) -> List[int]:
        """Returns the parent node for a given node (used for upward routing)."""
        if node_id in self.parent_of:
            return [self.parent_of[node_id]]
        return []

    def get_children(self, node_id: int) -> List[int]:
        """Returns all direct children of a given node (used for top-down broadcast)."""
        return self.children_of.get(node_id, [])

    def get_layer_peers(self, node_id: int) -> List[int]:
        """Returns same-layer ring neighbors for intra-layer gossip."""
        return self.layer_peers.get(node_id, [])

    def get_layer_for_node(self, node_id: int) -> int:
        """Returns the layer index a node belongs to."""
        return self.node_to_layer[node_id]

    def get_nodes_in_layer(self, layer_idx: int) -> List[int]:
        """Returns all node IDs in a given layer."""
        return self.layer_nodes.get(layer_idx, [])

    def get_server_connected_clients(self) -> List[int]:
        """Returns the direct children of the root server."""
        return self.children_of.get(self.root_id, [])

    def get_all_intermediate_ids(self) -> List[int]:
        """Returns all intermediate aggregator node IDs (layers 1 to L-2, excluding root)."""
        ids = []
        for layer_idx in range(1, len(self.layers) - 1):
            ids.extend(self.layer_nodes[layer_idx])
        return ids

    def get_depth(self) -> int:
        """Returns the number of aggregation layers (excluding the client layer)."""
        return len(self.layers) - 1
