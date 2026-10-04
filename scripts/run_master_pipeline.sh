#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo "Starting Master Pipeline on RTX 4090: $(date)"
echo "=========================================================="

cd /workspace/Topology-aware-FDL

# -----------------------------------------------------------------
# 1. 50-Client Scalability Benchmark (Table IV)
# -----------------------------------------------------------------
echo ">>> [1/2] Running 50-Client Scalability Benchmark..."
mkdir -p outputs/scale50

SEEDS=(42 123 7)
ALPHAS=("0.5" "0.1")

for alpha in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "--- Scale-50: alpha=${alpha}, seed=${seed} (40 rounds) ---"
    python3 scripts/run_scale_50clients.py --seed "$seed" --alpha "$alpha" --rounds 40 --force
  done
done

echo "--- Merging Scale-50 results ---"
python3 scripts/run_scale_50clients.py --merge

# -----------------------------------------------------------------
# 2. CIFAR-100 Ablation Study (Table VI)
# -----------------------------------------------------------------
echo ">>> [2/2] Running CIFAR-100 Ablation Study..."
mkdir -p outputs/baselines/ablation

python3 scripts/run_cifar100_ablation.py --rounds 25 --seeds 42 123 7 --force

# -----------------------------------------------------------------
# 3. Packaging Artifacts
# -----------------------------------------------------------------
echo ">>> Packaging artifacts into zip..."
mkdir -p /workspace/outputs
cd /workspace/Topology-aware-FDL
zip -r /workspace/outputs/scale50_and_ablation_artifacts.zip \
  outputs/scale50 \
  outputs/scale_50clients_results.json \
  outputs/baselines/ablation

echo "=========================================================="
echo "MASTER_RUN_COMPLETE: $(date)"
echo "=========================================================="
