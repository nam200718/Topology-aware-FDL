# Topology-aware-FDL (TopoFDL)

Topology-aware-FDL is a PyTorch-based federated learning simulator for studying how communication topology affects convergence and robustness under Byzantine behavior.

It trains a shared CNN on non-IID MNIST client partitions and compares multiple FL communication patterns under the same experiment settings.

## What this project includes

- **Topologies**: `star`, `ring`, `gossip`, `hierarchical`, `layered`
- **Engines**:
  - Centralized engine (star)
  - Decentralized engine (ring, gossip)
  - Hierarchical engine
  - Layered engine (with intra-layer gossip)
- **Byzantine modes**:
  - `label_flip`
  - `gradient_ascent`
  - `sign_flip`
  - `random_noise`
- **Outputs**:
  - Per-experiment `metrics.json`
  - Per-experiment `metrics.csv` (if `pandas` is available)
  - Matrix plot `outputs/matrix/convergence_matrix.png` for matrix runs

## Repository layout

```text
TopoFDL/
├── main.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data/dataset.py
│   ├── core/
│   └── topologies/
├── tests/
└── docs/
```

## Setup

From `/home/runner/work/Topology-aware-FDL/Topology-aware-FDL/TopoFDL`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (for tests):

```bash
pip install pytest
```

## Usage

### 1) Single default run

Runs a short star-topology smoke experiment (`smoke_test_mnist`) from `main.py`.

```bash
python main.py
```

### 2) Full matrix run

Sweeps all topologies against byzantine rates `0.0`, `0.1`, `0.3`.

```bash
python main.py --matrix
```

## Output locations

- Single/default run outputs: `outputs/<experiment_name>/`
- Matrix outputs: `outputs/matrix/`

Each experiment folder contains metrics history by round, including `test_accuracy`, `test_loss`, and participation counts.

## Testing

```bash
python -m pytest -q
```

## Documentation

- [docs/COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

## License

MIT — see [LICENSE](LICENSE).
