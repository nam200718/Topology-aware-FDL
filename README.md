# Topology-aware-FDL

A PyTorch-based federated learning simulator for studying how communication topology affects convergence and robustness under Byzantine client behavior.

## Repository layout

```text
Topology-aware-FDL/
├── README.md            ← this file
└── TopoFDL/             ← main package (all code and documentation)
    ├── main.py
    ├── requirements.txt
    ├── LICENSE
    ├── README.md
    ├── src/
    ├── tests/
    └── docs/
```

## Quickstart

```bash
git clone https://github.com/nam200718/Topology-aware-FDL.git
cd Topology-aware-FDL/TopoFDL
pip install -r requirements.txt
python main.py          # single star-topology smoke run (5 rounds, 5 clients)
python main.py --matrix # full topology × Byzantine-rate sweep
```

MNIST is downloaded automatically on the first run into `TopoFDL/data/`.

## Documentation

All detailed documentation lives inside `TopoFDL/`:

| File | Contents |
|------|----------|
| [`TopoFDL/README.md`](TopoFDL/README.md) | Feature overview, setup, and usage |
| [`TopoFDL/docs/ARCHITECTURE.md`](TopoFDL/docs/ARCHITECTURE.md) | Runtime flow, component responsibilities, ID conventions |
| [`TopoFDL/docs/COMPLETE_GUIDE.md`](TopoFDL/docs/COMPLETE_GUIDE.md) | Full technical reference (configuration, topologies, engines, attacks) |
| [`TopoFDL/docs/EXPERIMENTS.md`](TopoFDL/docs/EXPERIMENTS.md) | Experiment configurations and expected outputs |

## License

MIT — see [`TopoFDL/LICENSE`](TopoFDL/LICENSE).
