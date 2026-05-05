# Architecture Guide

FedlEARNING is designed to be a purely simulated, extensible environment for researching Topologies and Robustness models in Federated Learning.

## Core Hierarchy

The framework splits execution into specialized nodes:
- `BaseEngine`: Manages the round-robin logic, instantiates the local random vectors safely ensuring exact deterministic reproducibility.
- `CentralizedEngine`: Manages Server-to-Client architectures like `StarTopology` directly syncing central weights outward.
- `DecentralizedEngine`: Explicitly leverages physical node mapping where clients exclusively query their spatial neighbors per round, calculating local proxy gradients against aggregations of peers. Used for `RingTopology`, `GossipTopology`.
- `HierarchicalEngine`: Establishes cluster-head proxy nodes mapping nested Edge-to-Cloud distributions natively.
- `HierarchicalEnsembleEngine`: Implements a multi-tier ensemble approach where clients train and combine Root, Parent, and Local models for enhanced personalization and robustness.
- `LayeredEngine`: A recursive multi-layer aggregation engine that handles deep DAG structures with optional intra-layer gossip.

## The Interfaces

- `Topology`: Requires `build(num_clients, seed)` and `get_neighbors(node_id)`.
- `ClientState`: Encapsulates purely structural vector states along with identifying characteristics. Leverages PyTorch tensors for model weights.
- `Aggregator`: Policy for combining multiple client states (e.g., FedAvg).

## Training and Updating

The framework uses `PyTorchLocalUpdater` to perform local training on clients. It supports training multiple models simultaneously for ensemble-based topologies.

## Adding Topologies

To add a new topology (e.g. Tree or Scale-free), inherit `Topology`, enforce its geometric invariant inside `src/topologies/checks.py`, and link it inside `main.py#build_topology_and_engine`.
