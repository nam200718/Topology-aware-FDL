# Architecture Guide

FedlEARNING is designed to be a purely simulated, extensible environment for researching Topologies and Robustness models in Federated Learning.

## Core Hierarchy

The framework splits execution into specialized nodes:
- `BaseEngine`: Manages the round-robin logic, instantiates the local random vectors safely ensuring exact deterministic reproducibility.
- `CentralizedEngine`: Manages Server-to-Client architectures like `StarTopology` directly syncing central weights outward.
- `DecentralizedEngine`: Explicitly leverages physical node mapping where clients exclusively query their spatial neighbors per round, calculating local proxy gradients against aggregations of peers. Used for `RingTopology`, `GossipTopology`.
- `HierarchicalEngine`: Establishes cluster-head proxy nodes mapping nested Edge-to-Cloud distributions natively.

## The Interfaces

- `Topology`: Requires `build(num_clients, seed)` and `get_neighbors(node_id)`.
- `ClientState`: Encapsulates purely structural vector states along with identifying characteristics. Currently leverages purely mock approximate update calculations avoiding heavy PyTorch autograd logic intentionally.
- `FailureModel`: Wraps dropout and straggler simulation based on random intervals deterministically synchronized.

## Adding Topologies

To add a new topology (e.g. Tree or Scale-free), inherit `Topology`, enforce its geometric invariant inside `src/topologies/checks.py`, and link it inside `main.py#build_topology_and_engine`.
