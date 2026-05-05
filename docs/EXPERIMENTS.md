# Experiment Cookbook & Interpretation Guide

## Running Baseline

To generate a single baseline simulation spanning $100$ clients across $50$ rounds acting purely on a Star (FedAvg) projection:

```bash
python main.py
```

This populates `./outputs/smoke_test_baseline/metrics.json`.

## Generating Core Matrices

To automatically sweep Topology combinations (`Star`, `Ring`, `Gossip`, `Hierarchical`, `Hierarchical Ensemble`, `Layered`) against systemic Byzantine threat models (`0%`, `10%`, `30%`), use:

```bash
python main.py --matrix
```

This exports individual JSON reports per topology+robustness vector, dropping `convergence_matrix.png` directly into `./outputs/matrix`.

## Topology Comparison Study

To compare all implemented topologies (Star, Ring, Gossip, Hierarchical, and Hierarchical Ensemble) in a single study:

```bash
python3 scripts/run_comparison.py
```

This will run each topology for 10 rounds and generate a combined accuracy chart in `./outputs/comparison_study/topology_comparison.png`.

## Hierarchical Ensemble Experiment

The Hierarchical Ensemble is a specialized topology where clients train three models (Root, Parent, Local) and use an ensemble for inference. To run a standalone test:

```bash
python3 scripts/run_ensemble_experiment.py
```

## Layered Topology Experiment

The Layered topology can be configured with varying depths and gossip steps. To run a complex layered experiment:

```bash
python main.py --matrix # This includes layered by default
```

Or configure manually in `main.py`.

## Interpreting Research Claims

1. **Convergence (L2 Distance)**: Plotted curves detail iterative movement toward `global_target`. Purely simulated vector bounds map `0` as "Perfect knowledge of data target". 
2. **Topology Resilience**:
   - Centralized Topologies (Star) naturally yield faster gradient transmission (depth 1 to any node) yielding sharper descent curves.
   - P2P Topologies (Ring/Gossip) structurally require multiple hops to distribute information globally. You will strictly observe geometric latency relative to their average shortest path metrics. Ring converges the slowest.
   - Byzantine Threat Response: Unweighted pure `FedAvg` aggregations utilized mechanically break rapidly near $>25\%$ Byzantine flipped participation. Expect sharp L2 divergence on the generated matrices directly correlating to this unhandled attack vector. Proposing Byzantine-resilient aggregators (e.g., Krum, Median) is designated for future research implementation hooks!
