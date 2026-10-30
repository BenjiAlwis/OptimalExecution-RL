#!/usr/bin/env bash
# run_calibrated.sh — week-7 RL runs on the calibrated environments.
#
# Settings are the ones weeks 4-7 established, not defaults:
#   --mode clip        adaptive KL matches clip on central performance but
#                      blew up 1 checkpoint in 5 on a cliff-shaped objective
#                      (report section 6.6); clip had zero blow-ups.
#   --gamma 1.0        the execution objective is undiscounted, finite-horizon.
#   --no-clip-vloss    value clipping throttles a critic whose targets are far
#                      from its initialisation (report section 4.5).
#   --tag cal          keeps these runs in their own plot series.
# make_env must gate its normalizer block off for these env ids (they start
# with "OE-"), and train() must save periodic checkpoints.
#
# Usage:
#   bash run_calibrated.sh                       # headline set, 1M steps
#   bash run_calibrated.sh 2000000 3             # 2M steps, 3 concurrent
#   OE_ENVS="OE-GOOG-kT8-v0" bash run_calibrated.sh 1000000 1
set -e

TIMESTEPS="${1:-1000000}"
PARALLEL="${2:-3}"
PYTHON="${PYTHON:-python3}"
SEEDS=${OE_SEEDS:-"1 2 3"}

# Headline set: the concave-book names at high urgency, where the tilted-AC
# bound is largest (12.4% AMZN, 12.2% GOOG) and a learned policy has the most
# room to beat a one-parameter rule; plus one tick-constrained control.
ENVS=${OE_ENVS:-"OE-AMZN-kT4-v0 OE-AMZN-kT8-v0 OE-GOOG-kT4-v0 OE-GOOG-kT8-v0 OE-INTC-kT4-v0"}

if [ -d /workspace ]; then LOG_DIR="/workspace/runs"; else LOG_DIR="runs"; fi

echo "=== week-7 calibrated RL: ${TIMESTEPS} steps, ${PARALLEL} concurrent ==="
echo "envs:  ${ENVS}"
echo "seeds: ${SEEDS}"
echo "AC scores exactly -1.000 in every env; episodic_return > -1 beats AC."
echo "tilted-AC bounds to beat: AMZN kT4 12.00%, kT8 12.36%;"
echo "                          GOOG kT4 11.27%, kT8 12.17%; INTC kT4 4.86%"
echo "=============================================================="

for env in ${ENVS}; do
  for seed in ${SEEDS}; do
    echo ">>> ${env} | clip | seed ${seed}"
    "${PYTHON}" ppo_continuous.py \
      --env-id "calibrated_gym:${env}" \
      --mode clip --seed "${seed}" \
      --total-timesteps "${TIMESTEPS}" \
      --gamma 1.0 --no-clip-vloss \
      --tag cal --log-dir "${LOG_DIR}" &
    while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do sleep 5; done
  done
done
wait
echo "=== complete. Evaluate: python evaluate_calibrated.py ==="
