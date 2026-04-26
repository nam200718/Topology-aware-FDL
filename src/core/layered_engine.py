import torch
from src.core.base_engine import BaseEngine
from src.core.interfaces import ClientState


class LayeredEngine(BaseEngine):
    """
    Engine for the Layered (Neural Network-like) topology with intra-layer gossip.
    
    Each round executes:
      1. Top-down broadcast: Server -> intermediate nodes -> clients
      2. Local training: Only leaf clients (layer 0) run PyTorch SGD
      3. Layer-by-layer bottom-up aggregation with gossip:
         For each layer (0, 1, ..., L-2):
           a. Run gossip_steps rounds of intra-layer peer averaging
           b. Aggregate children into parent nodes of the next layer
      4. Server receives final result
    """

    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)

        # Global server weights
        self.server_weights = self.clients_state[0].weights.clone()

        # Create ClientState for all non-client nodes (intermediate + root)
        self.intermediate_states = {}
        for layer_idx in range(1, len(self.topology.layers)):
            for node_id in self.topology.get_nodes_in_layer(layer_idx):
                self.intermediate_states[node_id] = ClientState(
                    node_id, self.server_weights.clone()
                )

    def _get_state(self, node_id: int) -> ClientState:
        """Retrieve the state for any node (client or intermediate)."""
        if node_id >= 0:
            return self.clients_state[node_id]
        return self.intermediate_states[node_id]

    def _set_weights(self, node_id: int, weights: torch.Tensor):
        """Set weights for any node."""
        if node_id >= 0:
            self.clients_state[node_id].weights = weights
        else:
            self.intermediate_states[node_id].weights = weights

    def _broadcast_downward(self, node_id: int, weights: torch.Tensor):
        """Recursively broadcast weights from a node down to all descendants."""
        children = self.topology.get_children(node_id)
        for child_id in children:
            self._set_weights(child_id, weights.clone())
            if child_id < 0:
                self._broadcast_downward(child_id, weights)

    def _run_intra_layer_gossip(self, layer_idx: int):
        """
        Run gossip_steps rounds of peer averaging within a single layer.
        Uses a sync buffer to prevent order-of-operation bias.
        """
        nodes = self.topology.get_nodes_in_layer(layer_idx)
        if len(nodes) <= 1:
            return  # No peers to gossip with

        gossip_steps = self.topology.gossip_steps
        for _ in range(gossip_steps):
            # Read all current weights into a buffer
            buffers = {}
            for node_id in nodes:
                peers = self.topology.get_layer_peers(node_id)
                if not peers:
                    buffers[node_id] = self._get_state(node_id).weights.clone()
                    continue

                # Average own weights with all peers' weights
                peer_states = [self._get_state(p) for p in peers]
                all_states = peer_states + [self._get_state(node_id)]
                buffers[node_id] = self.aggregator.aggregate(all_states)

            # Commit buffered weights simultaneously
            for node_id, weights in buffers.items():
                self._set_weights(node_id, weights)

    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))
        num_layers = len(self.topology.layers)

        # 1. Top-down broadcast: server weights flow to all layers
        self._broadcast_downward(self.topology.root_id, self.server_weights)

        # 2. Local training: only leaf clients (layer 0) perform PyTorch SGD
        for client_id in all_clients:
            state = self.clients_state[client_id].copy()
            from src.data.dataset import ClientDataset
            client_ds = ClientDataset(
                self.train_dataset, self.client_indices[client_id]
            )
            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
            )
            self.clients_state[client_id] = updated_state

        # 3. Layer-by-layer bottom-up: gossip then aggregate
        for layer_idx in range(num_layers - 1):
            # 3a. Intra-layer gossip at this layer
            self._run_intra_layer_gossip(layer_idx)

            # 3b. Aggregate children into parent nodes of the next layer
            parent_layer_idx = layer_idx + 1
            parent_nodes = self.topology.get_nodes_in_layer(parent_layer_idx)

            for parent_id in parent_nodes:
                children = self.topology.get_children(parent_id)
                if children:
                    child_states = [self._get_state(c) for c in children]
                    agg_weights = self.aggregator.aggregate(child_states)
                    self._set_weights(parent_id, agg_weights)

        # 4. Server receives the root's aggregated weights
        self.server_weights = self._get_state(self.topology.root_id).weights.clone()

        # 5. Metrics
        acc, test_loss = self.evaluate_model(self.server_weights)

        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": len(all_clients),
            "aggregation_depth": self.topology.get_depth(),
            "gossip_steps": self.topology.gossip_steps,
        }
        
        if getattr(self.config.clients, "use_ensemble", False):
            ens_acc, ens_loss = self.evaluate_ensemble()
            round_data["ensemble_test_accuracy"] = ens_acc
            round_data["ensemble_test_loss"] = ens_loss
            
        self.metrics.log_round(round_data)
