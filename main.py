import os
import sys

# --- Virtual environment check ---
_in_venv = os.environ.get("VIRTUAL_ENV") or (sys.prefix != sys.base_prefix)
if not _in_venv:
    try:
        import yaml, torch  # If dependencies are available in system python, allow execution
    except ImportError:
        _project_root = os.path.dirname(os.path.abspath(__file__))
        print("ERROR: Virtual environment is not activated.", file=sys.stderr)
        if os.name == "nt":
            print("  PowerShell:  .\\.venv\\Scripts\\Activate.ps1", file=sys.stderr)
            print("  Cmd:         .\\.venv\\Scripts\\activate.bat", file=sys.stderr)
            print(f"  Or run:      $env:VIRTUAL_ENV=\"1\"; python {' '.join(sys.argv)}", file=sys.stderr)
        else:
            print(f"  Run:         source {_project_root}/.venv/bin/activate", file=sys.stderr)
        sys.exit(1)

from src.config import SimulationConfig
from src.experiments.builder import build_topology_and_engine, check_invariants
from src.experiments.runner import SingleExperimentRunner, ExperimentRunner
from src.experiments.cli import main as cli_main


def run_experiment(config: SimulationConfig):
    """Run a single experiment configuration and return metrics history."""
    runner = SingleExperimentRunner(config)
    return runner.run()


def run_from_config(config_path: str):
    """Load a YAML config file or preset and run the experiment suite."""
    runner = ExperimentRunner.from_yaml(config_path)
    return runner.run()


if __name__ == "__main__":
    cli_main()
