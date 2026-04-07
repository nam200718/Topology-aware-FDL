# Architecture

## High-level runtime

`main.py` is the orchestrator:
1. Build `(topology, engine)` from `SimulationConfig.topology.type`
2. Build a `FedAvgAggregator`
3. Select device (`cuda` or `cpu`)
4. Construct engine and run topology invariants
5. Execute `engine.run()`

Topology-to-engine mapping:
- `star` -> `CentralizedEngine`
- `ring` -> `DecentralizedEngine`
- `gossip` -> `DecentralizedEngine`
- `hierarchical` -> `HierarchicalEngine`
- `layered` -> `LayeredEngine`

## Project structure

- `src/config.py`: Pydantic config models (`SimulationConfig`, `TopologyConfig`, `ClientConfig`, `EnvironmentConfig`, `RobustnessConfig`)
- `src/core/interfaces.py`: core interfaces (`Topology`, `Aggregator`) and `ClientState`, `MetricsCollector`
- `src/core/base_engine.py`: shared MNIST-based engine setup and evaluation
- `src/core/{centralized,decentralized,hierarchical,layered}_engine.py`: topology-specific round logic
- `src/core/aggregator.py`: weighted FedAvg aggregation (`data_samples`-weighted)
- `src/core/updater.py`: local PyTorch training + Byzantine attack transforms
- `src/core/model.py`: `SimpleCNN` for MNIST
- `src/data/dataset.py`: MNIST loading and non-IID client partitioning
- `src/topologies/*.py`: communication graph/topology implementations
- `src/topologies/checks.py`: invariant checks used after topology build

## BaseEngine responsibilities

All active engines inherit from `BaseEngine`:
- build topology with `topology.build(num_clients, seed)`
- download/partition MNIST (`partition_data_non_iid`)
- create initial flattened model weights from `SimpleCNN`
- initialize per-client `ClientState`
- sample Byzantine clients once from `robustness.byzantine_rate`
- run per-round loop and persist metrics (`metrics.json`, `metrics.csv` when `pandas` exists)

## Round behavior by engine

### CentralizedEngine (Star)
- Server broadcasts current global weights to all clients
- Clients train locally on their partition
- Server aggregates all returned states with FedAvg
- Metrics are computed on server weights

### DecentralizedEngine (Ring/Gossip)
- Each client averages neighbors + self (synchronous buffered exchange)
- Each client trains locally from buffered weights
- Metrics are computed on mean client weights

### HierarchicalEngine
- Server broadcasts to cluster heads
- Clients receive from assigned head and train locally
- Head-level aggregation inside each cluster
- Server aggregates head states

### LayeredEngine
- Recursive top-down broadcast from root through intermediate layers
- Local training at leaf clients only
- Bottom-up aggregation layer by layer
- Optional intra-layer gossip (`gossip_steps`) before upward aggregation

## Topology contracts and IDs

All topologies implement:
- `build(num_clients, seed)`
- `get_neighbors(node_id)`
- `get_server_connected_clients()`

ID conventions:
- client IDs: non-negative (`0..num_clients-1`)
- star server ID: `-1`
- hierarchical cluster-head IDs: negative (`-2, -3, ...`)
- layered intermediate/root IDs: negative (root is the single last-layer node)

## Byzantine handling

Byzantine behavior is implemented in `PyTorchLocalUpdater`:
- `label_flip`: `labels = 9 - labels`
- `gradient_ascent`: negate loss before backward pass
- `sign_flip`: negate final model vector
- `random_noise`: replace final model vector with Gaussian noise (`* 10`)

## Notes

- `src/core/engine.py` contains an older lightweight vector simulation engine and is not used by `main.py`.
- Invariants are enforced immediately after engine construction via `src/topologies/checks.py`.
