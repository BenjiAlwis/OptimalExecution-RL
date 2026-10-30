#!/usr/bin/env bash
# run_execution.sh
# Objective-mode grid for the optimal-execution environment, mirroring
# run_env.sh from the ppo_repro capstone: 3 modes x 3 seeds = 9 runs.
#
# Drop this file plus execution_env.py, baselines.py, execution_gym.py into
# the ppo_repro repo root (next to ppo_continuous.py). No changes to
# ppo_continuous.py are needed: the module-prefix env id makes gym.make
# import execution_gym, which registers OptimalExecution-v0.
#
# Usage:
#   bash run_execution.sh                # 500k steps each
#   bash run_execution.sh 50000          # quick smoke test
#   bash run_execution.sh 500000 3       # 3 runs at a time

set -e

TIMESTEPS="${1:-1000000}"
PARALLEL="${2:-1}"
PYTHON="${PYTHON:-python3}"

ENV_ID="execution_gym:OptimalExecution-v0"

if [ -d /workspace ]; then LOG_DIR="/workspace/runs"; else LOG_DIR="runs"; fi

MODES=("clip" "noclip" "kl")
SEEDS=(1 2 3)

echo "=== OptimalExecution-v0 grid ==="
echo "timesteps: ${TIMESTEPS}   parallel: ${PARALLEL}   log dir: ${LOG_DIR}"
echo "analytical references (episodic return): AC = -3.2416, TWAP = -3.2687"
echo "================================"

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ">>> ${ENV_ID} | ${mode} | seed ${seed}"
    "${PYTHON}" ppo_continuous.py \
      --env-id "${ENV_ID}" \
      --mode "${mode}" \
      --seed "${seed}" \
      --total-timesteps "${TIMESTEPS}" \
      --gamma 1.0 --no-clip-vloss \
      --log-dir "${LOG_DIR}" &
    while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do
      sleep 5
    done
  done
done

wait
echo "=== complete; generating figures ==="
"${PYTHON}" plot_results.py --log-dir "${LOG_DIR}" --out figures
echo "=== done. Optimum: episodic return -> -3.2416 (AC). ==="
