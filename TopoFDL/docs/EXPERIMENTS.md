# Experiments

This project provides two entrypoints from `TopoFDL/main.py`.

## 1) Single default run

```bash
python main.py
```

Default config:
- `experiment_name="smoke_test_mnist"`
- `num_rounds=5`
- topology: `star`
- clients: `num_clients=5`, `local_lr=0.01`, `local_steps=1`
- robustness: defaults (`byzantine_rate=0.0`, `byzantine_type="label_flip"`)

Outputs:
- `./outputs/smoke_test_mnist/metrics.json`
- `./outputs/smoke_test_mnist/metrics.csv` (if `pandas` is installed)

Notes:
- MNIST is downloaded on demand into `./data`.
- Device is auto-selected (`cuda` if available, otherwise `cpu`).

## 2) Matrix run

```bash
python main.py --matrix
```

This runs a fixed sweep over:
- topologies: `star`, `ring`, `gossip`, `hierarchical`, `layered`
- byzantine rates: `0.0`, `0.1`, `0.3`

Per-run config in the matrix:
- `num_rounds=5`
- `num_clients=10`
- `local_lr=0.01`
- `local_steps=2`
- `seed=42`
- layered-only params: `layers=[10, 4, 2, 1]`

Outputs:
- Per experiment: `./outputs/matrix/<topology>_byz_<rate>/metrics.json` and `metrics.csv` (if `pandas` is installed)
- Aggregate figure: `./outputs/matrix/convergence_matrix.png`

## Metrics you will see

Common round fields:
- `round`
- `test_accuracy`
- `test_loss`
- `participating_clients`
- `total_clients_targeted`

Layered topology also logs:
- `aggregation_depth`
- `gossip_steps`

## Byzantine behavior in experiments

Byzantine clients are sampled once at engine initialization using `byzantine_rate`, then keep their assigned attack type (`byzantine_type`) for all rounds.

Supported attack modes:
- `label_flip` (labels transformed as `9 - label` during local training)
- `gradient_ascent` (loss sign inverted before backprop)
- `sign_flip` (post-training model update multiplied by `-1`)
- `random_noise` (post-training update replaced by high-variance random noise)
