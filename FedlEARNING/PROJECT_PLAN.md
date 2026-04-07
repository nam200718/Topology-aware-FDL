# Federated Learning Topology Research Simulator Plan

Date: 2026-04-03

## Goal
Build a Python program that benchmarks federated learning topologies for performance and robustness in a pure-simulation setting (approximate training), targeting up to 100 clients with local JSON/CSV outputs and publication-ready plots.

## Confirmed Decisions
- Language: Python
- v1 execution model: Pure simulation (approximate training)
- Initial topologies: Star (FedAvg), Ring, Hierarchical (2-level), Gossip P2P
- Early scale target: Up to 100 clients
- Tracking: Local JSON/CSV plus plots
- Out of scope for v1: Real distributed deployment and 1000+ client optimization

## Phased Plan

### Phase 1 - Foundation and Baseline
1. Initialize project structure, dependency management, config system, and reproducibility utilities.
2. Define core interfaces:
   - Client state
   - Topology
   - Aggregator
   - Scheduler
   - Failure model
   - Metrics collector
   - Simulation engine
3. Implement pure-simulation engine MVP:
   - Round-based simulation
   - Approximate local update dynamics
   - Deterministic RNG streams
   - Virtual communication cost accounting
4. Implement Star/FedAvg baseline and run smoke benchmark.
5. Validate output logging end-to-end.

### Phase 2 - Topologies and Robustness
1. Add Ring, Hierarchical, and Gossip topologies.
2. Add topology invariant checks:
   - Connectivity
   - Neighbor constraints
   - Deterministic graph seeding
3. Implement robustness injection modules:
   - Stragglers
   - Dropout/churn
   - Byzantine clients
   - Partial connectivity
   - Delay and bandwidth perturbations
4. Create experiment matrix across topology x failure x non-IID severity.
5. Implement metrics:
   - Convergence metrics
   - Communication metrics
   - Robustness metrics
   - Fairness metrics

### Phase 3 - Analysis, Validation, and Handoff
1. Export local JSON/CSV summaries.
2. Generate plots:
   - Convergence curves
   - Heatmaps
   - Sensitivity charts
3. Add tests:
   - Unit tests for topology/failure/metric correctness
   - Integration tests for deterministic replay
4. Write documentation:
   - Architecture guide
   - Experiment cookbook
   - Interpretation guide for research claims

## Verification Checklist
1. Determinism: Same seed and config produce identical outputs.
2. Topology sanity: Invariants hold for all four topologies.
3. Failure calibration: Observed rates match configured rates within tolerance.
4. Metric integrity: Communication and convergence metrics are internally consistent.
5. Matrix completeness: Every experiment cell generates outputs.
6. Robustness trends: Degradation under harsher conditions is reproducible and sensible.

## Recommended Technical Choices
1. Approximate training model:
   - Option A: Scalar loss dynamics (fastest)
   - Option B: Lightweight vector update model (more realistic)
   - Recommendation: Option B
2. Byzantine attack coverage:
   - Include label-flip, sign-flip, and random-noise attacks
3. Experiment budget control:
   - Use fractional or sampled design if full factorial grows too large

## Suggested Immediate Next Steps
1. Create initial Python package skeleton and folders.
2. Add minimal config and experiment runner entry point.
3. Implement Star/FedAvg simulation path first.
4. Run first smoke experiment and save baseline JSON.
