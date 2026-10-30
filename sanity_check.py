"""Sanity harness: verify the simulator against Almgren-Chriss closed forms.

Checks
------
1. TWAP: simulated mean and std of implementation shortfall match E[IS], sqrt(Var[IS]).
2. AC schedule: same check for the analytical optimal schedule.
3. Optimality: on the mean-variance objective J = E[IS] + lam * Var[IS], the AC
   schedule beats TWAP and beats random perturbations of itself.
4. lam -> 0 limit: AC schedule collapses to TWAP.

Run:  python sanity_check.py
Exits nonzero if any check fails, so it can sit in CI.
"""

from __future__ import annotations

import sys

import numpy as np

from execution_env import ACParams, ExecutionEnv
from baselines import twap_schedule, ac_schedule, SchedulePolicy, evaluate_policy

N_EPISODES = 20_000
Z_TOL = 4.0        # tolerance in standard errors for mean checks
STD_RTOL = 0.05    # relative tolerance for std checks

def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def simulate_schedule(p: ACParams, schedule: np.ndarray) -> np.ndarray:
    env = ExecutionEnv(p, continuous=True)
    return evaluate_policy(env, SchedulePolicy(schedule), n_episodes=N_EPISODES)


def main() -> int:
    p = ACParams()
    env = ExecutionEnv(p, continuous=True)
    all_ok = True

    for name, schedule in [("TWAP", twap_schedule(p)), ("AC optimal", ac_schedule(p))]:
        e_theory = env.expected_cost(schedule)
        sd_theory = np.sqrt(env.cost_variance(schedule))
        sf = simulate_schedule(p, schedule)
        se = sf.std(ddof=1) / np.sqrt(len(sf))
        z = abs(sf.mean() - e_theory) / se
        all_ok &= check(
            f"{name} mean cost",
            z < Z_TOL,
            f"sim {sf.mean():,.0f} vs theory {e_theory:,.0f}  (z = {z:.2f})",
        )
        rel = abs(sf.std(ddof=1) - sd_theory) / sd_theory
        all_ok &= check(
            f"{name} cost std",
            rel < STD_RTOL,
            f"sim {sf.std(ddof=1):,.0f} vs theory {sd_theory:,.0f}  (rel err = {rel:.3f})",
        )

    # --- optimality on the mean-variance objective
    def J(schedule: np.ndarray) -> float:
        return env.expected_cost(schedule) + p.lam * env.cost_variance(schedule)

    twap, ac = twap_schedule(p), ac_schedule(p)
    all_ok &= check(
        "AC beats TWAP on J = E + lam*V",
        J(ac) < J(twap),
        f"J_AC = {J(ac):,.0f} < J_TWAP = {J(twap):,.0f}",
    )

    rng = np.random.default_rng(0)
    beaten = 0
    for _ in range(200):
        pert = ac * rng.uniform(0.9, 1.1, size=ac.shape)
        pert *= p.X / pert.sum()  # keep total shares fixed
        beaten += J(pert) > J(ac)
    all_ok &= check(
        "AC beats 200 perturbed schedules",
        beaten == 200,
        f"{beaten}/200 perturbations are worse",
    )

    # --- lam -> 0 degenerates to TWAP
    p0 = ACParams(lam=1e-12)
    dev = np.max(np.abs(ac_schedule(p0) - twap_schedule(p0))) / (p0.X / p0.N)
    all_ok &= check("lam->0 recovers TWAP", dev < 1e-3, f"max rel deviation = {dev:.2e}")

    print("\nAll checks passed." if all_ok else "\nSOME CHECKS FAILED.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
