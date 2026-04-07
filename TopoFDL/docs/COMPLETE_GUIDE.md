# FedlEARNING — Comprehensive Technical Reference

> **Document Objective:** A comprehensive, formal technical reference detailing the architecture, execution flow, empirical design choices, and interpretation guidelines for the FedlEARNING PyTorch-based simulation framework.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Federated Learning Foundations in FedlEARNING](#2-federated-learning-foundations-in-fedlearning)
3. [Memory Serialization Strategy (O(1) Memory)](#3-memory-serialization-strategy-o1-memory)
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

FedlEARNING is an empirical, high-fidelity research simulator built to evaluate Federated Learning (FL) structural paradigms (topologies) and Byzantine fault tolerance. Utilizing the PyTorch backend, the system simulates distributed training of a real Convolutional Neural Network (`SimpleCNN`) on standard empirical datasets (e.g., MNIST).

Unlike rudimentary simulations relying on dummy mathematical abstractions, FedlEARNING conducts authentic Forward and Backward passes using Stochastic Gradient Descent (SGD) on highly non-IID data shards, while guaranteeing performance on consumer hardware through strict structural memory serialization policies.

---

## 2. Federated Learning Foundations in FedlEARNING

FedlEARNING is engineered to examine two principal research vectors in distributed machine learning:
1. **Network Topologies:** How structural communication constraints (e.g., Star/Hub-and-Spoke, Decentralized Rings, Random Gossip, or Hierarchical clusters) impact convergence rates.
2. **Robustness:** The systemic resilience against active Byzantine adversaries propagating intentionally poisoned gradients or corrupted parameters.

---

## 3. Memory Serialization Strategy (O(1) Memory)

Instantiating an independent `nn.Module` (a PyTorch model graph) for hundreds of decentralized clients instantly exceeds conventional desktop RAM capacities. FedlEARNING handles this computationally taxing process via memory serialization:

1. A singular, static `SimpleCNN` instance resides in the main simulation's memory.
2. The fundamental `ClientState` holds its localized weights simply as a flattened, detached 1D `torch.Tensor`.
3. When local training executes, the system actively injects the client's 1D tensor into the static CNN, unflattening it sequentially. Following local mini-batch updates, the parameters are subsequently flattened and extracted back into the `ClientState`.
4. This enforces an $O(1)$ scaling concerning memory overhead related to active parameter graphs, circumventing GPU/CPU memory overflows regardless of the client pool cardinality.

---

## 4. Data Heterogeneity and MNIST Sharding

To reliably mimic empirical FL bottlenecks, the simulator forces acute structural gradient drift through highly Non-IID data allocations. 

Via the `dataset.py` integration routines, torchvision datasets are downloaded and comprehensively sorted by class labels. The global corpus is fragmented into distinct shards. Participating clients are assigned a minimal subset of shards. Consequently, individual clients structurally optimize local parameters against a highly restricted subset of labels (e.g., client $C_i$ may only see images representing digits '3' and '7'), forcing systemic reconciliation at the aggregation phase.

---

## 5. Directory Structure

```text
FedlEARNING/
├── main.py                          ← Primary simulation execution router
├── requirements.txt                 ← Python environment definitions (torch, torchvision, etc.)
├── Project Context Files
│
├── src/                             
│   ├── config.py                    ← Pydantic environment configurations
│   ├── dataset.py                   ← Data fetching, caching, and Non-IID sharding protocols
│   ├── core/                        
│   │   ├── interfaces.py            ← Architectural abstractions for generic classes
│   │   ├── base_engine.py           ← Universal simulation initialization / target loops
│   │   ├── centralized_engine.py    ← Orchestration for Star frameworks
│   │   ├── decentralized_engine.py  ← Handling synchronous buffer aggregations (Ring/Gossip)
│   │   ├── hierarchical_engine.py   ← Cluster-head processing loops
│   │   ├── aggregator.py            ← `FedAvgAggregator` arithmetic consensus logic
│   │   ├── updater.py               ← `PyTorchLocalUpdater` tensor injection and SGD looping
│   │   ├── layered_engine.py        ← Recursive multi-layer aggregation engine
│   │   └── model.py                 ← CNN architecture definitions
│   │
│   ├── topologies/                  
│   │   ├── star.py                  ← Centralized hub topology
│   │   ├── ring.py                  ← 2-degree linear decentralized network
│   │   ├── gossip.py                ← Randomized regular graph connection handling
│   │   ├── hierarchical.py          ← Multi-tier grouping schemas
│   │   ├── layered.py               ← Neural network-like deep DAG topology
│   │   └── checks.py                ← Structural invariants asserting valid connections
│   │
│   └── utils/
│       └── random.py                ← Robust seed isolation and generic RNG protocols
│
├── tests/                           ← Automated validation sweeps (pytest compatibility required)
│
└── outputs/                         ← Matrix results, JSON trajectories, and Matplotlib `.png` results
```

---

## 6. Execution Guide

### Dependency Procurement
The simulation environment fundamentally relies upon standard deep learning utilities and matrix compute engines. 

```bash
cd c:\FedlEARNING
pip install -r requirements.txt
```

### Full matrix structural analysis
The standard benchmarking routine runs a permutation of all established topologies against varying levels of Byzantine interference simultaneously. Given the PyTorch integration, expect runs to span ~5-15 minutes processing authentic epochs.

```bash
python main.py --matrix
```

This procedure instantiates independent configurations mapping Star, Ring, Gossip, Hierarchical, and Layered frameworks over varying malicious presence concentrations ($0.0$, $0.1$, $0.3$), finalizing with a multi-panel Matplotlib `.png` graph plotting testing accuracy.

---

## 7. Configuration Parameters

The configuration architectures explicitly dictate empirical constraints natively through Pydantic strict typing models found internally at `src/config.py`.

### Client/Environment Parameters
- `num_clients`: Pool of locally executing end nodes. Modulating this variable directs training sample constraints explicitly. 
- `local_lr`: The hyperparameter for localized Stochastic Gradient Descent iterations (`0.01` baseline).
- `local_steps`: Standardized Epoch execution loops (specifically batched steps).

### System Threat Parameters
- `byzantine_rate`: The explicit saturation rate of corrupted clients ($0.0$ to $1.0$).
- `byzantine_type`: Structural definitions of node compromises. Currently supported:
  - `label_flip`: Attack targeting data labeling.
  - `sign_flip`: Attack targeting parameter weight signs.
  - `random_noise`: Attack targeting parameter integrity via Gaussian noise.
  - `gradient_ascent`: Active optimization sabotage through loss inversion.

---

## 8. Core Component Interfaces

- `ClientState`: Retains identity (`client_id`) alongside model representations maintaining mathematical weights strictly as 1-dimensional detached standard scalar arrays (native PyTorch tensors). 
- `Topology`: Formulates Graph theoretical foundations regulating permitted information transfer loops securely checking deterministic connectivity paradigms mapping adjacent identities strictly. 
- `Aggregator`: Formal routines evaluating multiple 1D target states, returning aggregated updates generally processing weighted arithmetic scaling schemas based uniformly on data counts. 

---

## 9. Network Topologies

1. **Star (Centralized)**: Single parameter server mapping uniformly to an extensive array of disconnected worker clients. High reliance strictly mapping client vectors concurrently to one hub. Evaluates theoretical upper limits.
2. **Ring**: Linear looping mapping forcing local clients to restrict parameters uniquely to two local neighbors iteratively.
3. **Gossip**: Parameter coordination restricting information arbitrarily using predetermined degrees. 
4. **Hierarchical**: Multi-tier architecture abstracting structural nodes to central cluster mapping vectors scaling locally prior to global amalgamation.
5. **Layered (Neural Network-like)**: A deep, multi-layer DAG topology generalizing the Hierarchical approach to arbitrary depth, with optional **intra-layer gossip** for lateral knowledge sharing. Configured via `layers` and `gossip_steps` parameters:

```python
topology=TopologyConfig(type="layered", params={"layers": [10, 4, 2, 1], "gossip_steps": 1})
```

```
Layer 0 (Clients):     C0 ←→ C1 ←→ C2    C3 ←→ C4 ←→ C5    C6 ←→ C7 ←→ C8 ←→ C9
                        \    |   /          |    |              \    |   /       |
Layer 1 (Sub-heads):       A0     ←——→      A1     ←——→          A2     ←——→    A3
                            \              /                      \            /
Layer 2 (Regional):           B0     ←——→                          B1
                               \                                 /
Layer 3 (Server):                          SERVER
```

   Vertical arrows (`\`, `/`, `|`) represent parent-child aggregation. Horizontal arrows (`←→`) represent **intra-layer gossip** — nodes sharing knowledge with their same-layer peers (connected in a ring) before propagating updates upward.

   **Round execution flow:**
   1. Top-down broadcast: Server weights flow downward through all layers.
   2. Local training: Only Layer 0 clients perform PyTorch SGD.
   3. For each layer (bottom-up): Run `gossip_steps` rounds of peer averaging, then aggregate children into parent nodes.
   4. Server receives the final aggregated result.

   **Byzantine defense**: The combination of vertical aggregation and horizontal gossip creates a double defense. A poisoned update is first diluted by peer gossip within its layer, then further diluted by aggregation at each subsequent layer. Setting `gossip_steps=0` disables lateral sharing (pure vertical aggregation).

---

## 10. Simulation Engines

Processing structures are inherently asynchronous when physically separated. Internally `FedlEARNING` applies explicitly robust synchronous iterations avoiding local variance. 
The **Decentralized Engine**, uniquely employs buffers structurally restricting local clients to read parallel neighbors safely without cascading sequential modifications skewing updates concurrently. 

---

## 11. Local PyTorch Training (`PyTorchLocalUpdater`)

This constitutes the principal execution cycle. Internally, models are trained utilizing generic Backpropagation procedures. 
1. The global flattened Tensor initiates updates reconstructing layers procedurally inside the static `nn.Module`.
2. Following standard dataloading on purely Non-IID shards utilizing native Mini-batch configurations evaluating strictly localized Cross-Entropy metrics.
3. Following local Epoch saturation, internal elements are processed yielding subsequent flattening. 

---

## 12. Byzantine Robustness and Adversarial Models

FedlEARNING supports several attack vectors to test the resilience of Federated Learning aggregation. These are injected during the local update phase in `PyTorchLocalUpdater`.

### Attack Vector Definitions:
- **Label Flipping (`label_flip`)**: A data poisoning attack where the client intentionally mislabels their local data before training. Specifically, labels are transformed via: `labels = 9 - labels` (turning '7s' into '2s', etc.).
- **Gradient Ascent (`gradient_ascent`)**: An active optimization sabotage where the client modifies the loss objective. Rather than minimizing cross-entropy, the client maximizes it: `loss = -criterion(outputs, labels)`. This forces the local model to drift away from the target class features.
- **Sign Flipping (`sign_flip`)**: A weight manipulation attack where the client performs honest training but inverts the signs of the resulting parameter vector before submission: `weights = -weights`.
- **Random Noise (`random_noise`)**: A denial-of-convergence attack where the client circumvents training and yields a high-variance randomized Gaussian tensor: `weights = torch.randn_like(weights) * 10.0`.

---

## 13. Aggregator Mechanics

Employs the standardized `FedAvg` (Federated Averaging) model structurally amalgamating 1D parameters equally evaluating basic arithmetic coefficients scaling relative to total dataset samples. Future evaluations targets replacing FedAvg with robust alternatives (e.g., Krum, Trimmed Mean, or Median) to mitigate the adversarial vectors described above.

---

## 14. Reproducibility and Stochasticity

Isolating local seed values structurally regulates global entropy ensuring absolute determinism universally. By mapping purely isolated `torch.Generator` elements localized explicitly processing unique client variations statically enforcing purely deterministic shard distributions across consecutive identical testing protocols universally resolving stochastic inconsistencies intrinsically. 

---

## 15. Empirical Evaluation Metrics

Formal result logging relies on testing explicitly defined datasets tracking global aggregation metrics mapping independently away from raw abstract distance values. Output CSVs represent standardized empirical parameters exclusively: 
1. **Test Accuracy (%)**: Tracking global CNN performance mapped natively utilizing 10,000 independent samples consistently verifying global aggregations correctly modeling true predictions. 
2. **Test Cross-Entropy Loss**: Standard structural loss verification universally mapping optimization descent. 

Tracking outputs visualize these matrices directly observing varying structural topology convergence variations evaluating true learning parameters fundamentally identifying systemic optimizations uniquely evaluating empirical baselines continuously measuring true generalized outputs uniformly. 

---

## 16. Common Research Questions

**Q: Can I integrate custom datasets (e.g., CIFAR-10)?**
Yes. Modify `dataset.py` mapping structural extraction processes routing localized shards tracking CIFAR natively utilizing standard `torchvision` integrations effectively updating structural shapes processing corresponding `SimpleCNN` input vectors accurately. 

**Q: Why does final accuracy fluctuate under 30% Byzantine presence?**
Without explicitly defined secure aggregations (e.g. Krum or Median), simple arithmetic averaging (`FedAvg`) integrates poisoned gradients unconditionally. Attacks like `label_flip` and `gradient_ascent` pull the global parameters towards a malicious or destructive objective, resulting in immediate performance decay.
