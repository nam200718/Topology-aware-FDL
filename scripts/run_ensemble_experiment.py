import os
import sys
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.experiments.runner import ExperimentRunner

DEFAULT_CONFIG = os.path.join(_project_root, "configs", "ensemble_experiment.yaml")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Ensemble Experiment")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to YAML config file")
    args = parser.parse_args()

    runner = ExperimentRunner.from_yaml(args.config)
    runner.run()


if __name__ == "__main__":
    main()
