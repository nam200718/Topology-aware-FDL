#!/usr/bin/env bash
# Setup Environment for Topology-aware-FDL (FedHEP)
set -e

echo "=========================================================="
echo "Setting up Python Environment for Topology-aware-FDL"
echo "=========================================================="

PYTHON_BIN=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.10+ is required but not found in PATH." >&2
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment $VENV_DIR already exists."
fi

# Activate venv
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Verifying PyTorch and Accelerator Support..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if cuda_avail:
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
elif mps_avail:
    print('Apple Silicon MPS acceleration available.')
else:
    print('Running on CPU.')
"

echo "=========================================================="
echo "Environment setup complete!"
echo "To activate:"
echo "  source $VENV_DIR/bin/activate"
echo "To verify test suite:"
echo "  pytest tests/ -q"
echo "=========================================================="
