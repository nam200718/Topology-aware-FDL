# Topology-aware-FDL (TopoFDL) — Comprehensive Technical Reference

> **Document Objective:** A comprehensive technical reference detailing the architecture, execution flow, empirical design choices, and interpretation guidelines for the Topology-aware-FDL PyTorch-based simulation framework.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Federated Learning Foundations in Topology-aware-FDL](#2-federated-learning-foundations-in-topology-aware-fdl)
3. [Memory Serialization Strategy](#3-memory-serialization-strategy)
4. [Data Heterogeneity and MNIST Sharding](#4-data-heterogeneity-and-mnist-sharding)
5. [Directory Structure](#5-directory-structure)
6. [Execution Guide](#6-execution-guide)
7. [Configuration Parameters](#7-configuration-parameters)
8. [Core Component Interfaces](#8-core-component-interfaces)
9. [Network Topologies](#9-network-topologies)
10. [Simulation Engines](#10-simulation-engines)
11. [Local PyTorch Training (`PyTorchLocalUpdater`)](#11-local-pytorch-training-pytorchlocalupdater)
12. [Byzantine Robustness and Adversarial Models](#12-byzantine-robustness-and-adversarial-models)
13. [Aggregator Mechanics](#13-aggregator-mechanics)
14. [Reproducibility and Stochasticity](#14-reproducibility-and-stochasticity)
15. [Empirical Evaluation Metrics](#15-empirical-evaluation-metrics)
16. [Common Research Questions](#16-common-research-questions)

---

## 1. Architectural Overview

Topology-aware-FDL (TopoFDL) is an empirical simulator for evaluating federated learning communication structures (topologies) and Byzantine robustness.

The framework performs real PyTorch training on a CNN (`SimpleCNN`) over non-IID MNIST client partitions, then compares how topology and adversarial behavior affect convergence.

---

## 2. Federated Learning Foundations in Topology-aware-FDL

TopoFDL is built to study two primary research axes:
1. **Network Topologies:** How communication structure (Star, Ring, Gossip, Hierarchical, Layered) affects learning dynamics.
2. **Robustness:** How Byzantine clients degrade training under poisoning or model corruption attacks.

---

## 3. Memory Serialization Strategy

The simulator stores each client model as a flattened 1D parameter tensor (`ClientState.weights`) and converts between vector and model parameters during local updates/evaluation.

1. `ClientState` keeps per-client model state as a flat tensor.
2. `PyTorchLocalUpdater` maps vector weights into `SimpleCNN` via `torch.nn.utils.vector_to_parameters`.
3. After local training, updated parameters are flattened back with `parameters_to_vector`.
4. This avoids keeping per-client `nn.Module` graphs resident in memory.

---

## 4. Data Heterogeneity and MNIST Sharding

To simulate realistic FL heterogeneity, TopoFDL uses highly non-IID partitioning.

In `src/data/dataset.py`, MNIST labels are sorted, split into shards, and shard groups are assigned per client. As a result, each client sees only a narrow subset of labels, creating client drift that must be reconciled by aggregation.

---

## 5. Directory Structure

```text
TopoFDL/
├── main.py                          ← Primary simulation entrypoint
├── requirements.txt                 ← Python dependencies
├── README.md
├── docs/
│   ├── COMPLETE_GUIDE.md
│   ├── ARCHITECTURE.md
│   └── EXPERIMENTS.md
├── src/
│   ├── config.py                    ← Pydantic simulation configuration models
│   ├── data/
│   │   └── dataset.py               ← MNIST loading + non-IID partitioning
│   ├── core/
│   │   ├── interfaces.py            ← Core abstractions + `ClientState`
│   │   ├── base_engine.py           ← Shared execution/evaluation pipeline
│   │   ├── centralized_engine.py    ← Star topology engine
│   │   ├── decentralized_engine.py  ← Ring/Gossip engine
│   │   ├── hierarchical_engine.py   ← Hierarchical engine
│   │   ├── layered_engine.py        ← Layered topology engine
│   │   ├── aggregator.py            ← `FedAvgAggregator`
│   │   ├── updater.py               ← `PyTorchLocalUpdater`
│   │   ├── model.py                 ← `SimpleCNN`
│   │   └── engine.py                ← Legacy/unused vector-engine prototype
│   ├── topologies/
│   │   ├── star.py
│   │   ├── ring.py
│   │   ├── gossip.py
│   │   ├── hierarchical.py
│   │   ├── layered.py
│   │   └── checks.py
│   └── utils/
│       └── random.py                ← Seeded random helpers
├── tests/
│   └── test_framework.py
└── outputs/                         ← Metrics and plots
```

---

## 6. Execution Guide

### Dependency installation

From the project root:

```bash
cd TopoFDL
pip install -r requirements.txt
```

### Single default run

```bash
python main.py
```

Default run config:
- experiment name: `smoke_test_mnist`
- topology: `star`
- rounds: `5`
- clients: `5`

### Full matrix analysis

```bash
python main.py --matrix
```

This sweeps topologies (`star`, `ring`, `gossip`, `hierarchical`, `layered`) across byzantine rates (`0.0`, `0.1`, `0.3`) and produces a matrix plot at `outputs/matrix/convergence_matrix.png`.

---

## 7. Configuration Parameters

Configuration models are defined in `src/config.py`.

### Client / environment parameters
- `num_clients`: Number of participating clients.
- `local_lr`: Local SGD learning rate.
- `local_steps`: Number of local training epochs per round.
- `env.seed`: Global seed used for topology/data partition reproducibility.
- `env.output_dir`: Base output directory.

### Topology parameters
- `topology.type`: One of `star`, `ring`, `gossip`, `hierarchical`, `layered`.
- `topology.params.degree_k`: Degree for gossip regular graph.
- `topology.params.num_clusters`: Number of clusters for hierarchical topology.
- `topology.params.layers`: Layer sizes for layered topology (first must equal `num_clients`, last must be `1`).
- `topology.params.gossip_steps`: Intra-layer gossip rounds for layered topology.

### Threat model parameters
- `byzantine_rate`: Fraction of clients marked Byzantine.
- `byzantine_type`: One of:
  - `label_flip`
  - `sign_flip`
  - `random_noise`
  - `gradient_ascent`

---

## 8. Core Component Interfaces

- `ClientState`: Stores `client_id`, flattened `weights`, metadata (`data_samples`), and byzantine flags.
- `Topology`: Defines graph behavior via `build`, `get_neighbors`, and `get_server_connected_clients`.
- `Aggregator`: Defines `aggregate(states)` behavior.
- `MetricsCollector`: Collects per-round dictionaries for JSON/CSV export.

---

## 9. Network Topologies

1. **Star (Centralized):** One server connected to all clients.
2. **Ring:** Each client communicates with two neighbors in a cycle.
3. **Gossip:** Clients communicate over a connected random regular graph.
4. **Hierarchical:** Clients are assigned to cluster-head nodes; heads aggregate upward to a server.
5. **Layered (Neural-network-like DAG):** Multi-layer hierarchy with optional same-layer gossip.

Layered example:

```python
topology=TopologyConfig(type="layered", params={"layers": [10, 4, 2, 1], "gossip_steps": 1})
```

Round flow in layered topology:
1. Top-down broadcast from root to descendants.
2. Local client training at layer 0.
3. For each layer bottom-up: intra-layer gossip, then child-to-parent aggregation.
4. Root state becomes new global model.

---

## 10. Simulation Engines

Engines implement synchronous round-based simulation:

- `CentralizedEngine`: Server broadcasts, clients train, server aggregates.
- `DecentralizedEngine`: Clients mix neighbor states through buffered synchronous aggregation, then train locally.
- `HierarchicalEngine`: Server ↔ heads ↔ clients with two-stage aggregation (cluster then global).
- `LayeredEngine`: Recursive multi-layer broadcast/aggregation with configurable intra-layer gossip.

---

## 11. Local PyTorch Training (`PyTorchLocalUpdater`)

Local update pipeline:
1. Reconstruct `SimpleCNN` from flattened client weights.
2. Train on the client-specific MNIST shard using SGD and cross-entropy.
3. Apply Byzantine behavior if enabled.
4. Flatten weights back and return updated `ClientState`.

---

## 12. Byzantine Robustness and Adversarial Models

Byzantine behavior is injected during/after local training in `PyTorchLocalUpdater`.

### Attack definitions
- **Label Flipping (`label_flip`)**: transforms labels as `labels = 9 - labels` before loss computation.
- **Gradient Ascent (`gradient_ascent`)**: inverts objective via `loss = -loss` to maximize error.
- **Sign Flipping (`sign_flip`)**: flips trained weight vector sign before upload.
- **Random Noise (`random_noise`)**: replaces outgoing vector with high-variance Gaussian noise.

---

## 13. Aggregator Mechanics

Aggregation uses `FedAvgAggregator` (`src/core/aggregator.py`).

- Computes weighted average of client weights by `data_samples`.
- Falls back to simple mean if total sample count is zero.

---

## 14. Reproducibility and Stochasticity

Reproducibility is controlled primarily by `env.seed`, which is used in topology construction and non-IID data partitioning.

The framework relies on deterministic seeded random-state helpers in `src/utils/random.py` and explicit seed setting in `main.py`.

---

## 15. Empirical Evaluation Metrics

Current experiments track model quality metrics, not vector L2 proxy distance.

Per-round outputs include:
1. **Test Accuracy (%)**
2. **Test Cross-Entropy Loss**
3. Participation counts (`participating_clients`, `total_clients_targeted`)
4. Layered-only metadata (`aggregation_depth`, `gossip_steps`)

Metrics are written to:
- `metrics.json` (always)
- `metrics.csv` (when `pandas` is installed)

---

## 16. Common Research Questions

**Q: Can I integrate custom datasets (e.g., CIFAR-10)?**  
Yes. Extend the data loading and partitioning logic in `src/data/dataset.py`, and ensure the model architecture in `src/core/model.py` matches the new input shape/task.

**Q: Why does accuracy degrade at higher Byzantine rates (e.g., 30%)?**  
Because the current global aggregation is standard FedAvg, which is not Byzantine-robust. Strong poisoning attacks can significantly skew the averaged update.
