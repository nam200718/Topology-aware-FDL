import os
import sys
import argparse

# Add the project root to the path so we can import from src and main
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)

# --- Virtual environment check ---
if not os.environ.get("VIRTUAL_ENV") and sys.prefix == sys.base_prefix:
    print("ERROR: Virtual environment is not activated.", file=sys.stderr)
    print(f"  Run:  source {_project_root}/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

from main import run_from_config

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ensemble_experiment.yaml")

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Ensemble Experiment")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to YAML config file")
    args = parser.parse_args()

    run_from_config(args.config)

if __name__ == "__main__":
    main()
