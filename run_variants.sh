#!/usr/bin/env bash
# run_variants.sh — week-5 training grid over the three AC-breaking variants.
# Uses the settings validated in week 4 (--gamma 1.0 --no-clip-vloss) and the
# kl objective (best mode in the week-4 ablation; override MODES to widen).
#
# Usage:
#   bash run_variants.sh                 # 1M steps each, 9 runs
#   bash run_variants.sh 100000          # smoke test
#   bash run_variants.sh 1000000 3       # 3 concurrent
set -e
TIMESTEPS="${1:-1000000}"
PARALLEL="${2:-1}"
PYTHON="${PYTHON:-python3}"
if [ -d /workspace ]; then LOG_DIR="/workspace/runs"; else LOG_DIR="runs"; fi

ENVS=("execution_gym:OptimalExecutionOU-v0" \
      "execution_gym:OptimalExecutionSqrt-v0" \
      "execution_gym:OptimalExecutionPressure-v0")
MODES=("kl")
SEEDS=(1 2 3)

echo "=== week-5 variant grid: ${TIMESTEPS} steps, ${PARALLEL} concurrent ==="
echo "targets (analyze_variants.py):"
echo "  OU:       BEAT  AC (J=3,627,452 MC) by >7.75% (tilted-AC bound)"
echo "  Sqrt:     MATCH numeric open-loop opt J=3,202,121 (AC +0.735% above)"
echo "  Pressure: MATCH AC (J=3,287,702; numeric opt coincides with AC)"

for env in "${ENVS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ">>> ${env} | ${mode} | seed ${seed}"
      "${PYTHON}" ppo_continuous.py \
        --env-id "${env}" --mode "${mode}" --seed "${seed}" \
        --total-timesteps "${TIMESTEPS}" --gamma 1.0 --no-clip-vloss \
        --log-dir "${LOG_DIR}" &
      while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do sleep 5; done
    done
  done
done
wait
echo "=== variant grid complete. Evaluate with: python evaluate_variant_policy.py ==="
