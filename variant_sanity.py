"""Sanity harness for the week-5 environment variants.

Tier 1 — exact degeneracy (the strongest possible check): with its knob
zeroed, each variant must produce BIT-IDENTICAL rewards and prices to the
validated base environment, same seed, same action sequence. This works
because variant-specific randomness lives on a separate state_rng, leaving
the price-noise stream untouched.

Tier 2 — closed-form checks where the variant admits one:
  - SqrtImpactEnv.expected_cost_sqrt and VolumePressureEnv.expected_cost_pressure
    are validated against Monte Carlo (deterministic schedules, noisy prices).
  - StochasticLiquidityEnv calibration: E[eta_t] = eta, so for any
    deterministic schedule the expected cost equals the BASE closed form —
    checked by MC. (This is the property that makes AC exactly optimal
    open-loop in this variant.)

Tier 3 — mechanism/directional checks:
  - liquidity: trading when z is low (deep liquidity) is cheaper than when high.
  - sqrt: TWAP temporary cost matches the base model (calibration point);
    large single trades are relatively cheaper than under quadratic cost.
  - pressure: bursty schedules cost more than spaced ones with equal volume.

Run: python variant_sanity.py   (exits nonzero on failure)
"""

from __future__ import annotations

import sys

import numpy as np

from execution_env import ACParams, ExecutionEnv
from execution_env_variants import (
    StochasticLiquidityEnv, SqrtImpactEnv, VolumePressureEnv,
)
from baselines import twap_schedule, SchedulePolicy, evaluate_policy

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def rollout(env, actions, seed):
    obs = env.reset(seed=seed)
    rews, prices = [], []
    for a in actions:
        obs, r, done, info = env.step(a)
        rews.append(r)
        prices.append(env.S)
    return np.array(rews), np.array(prices), info


def tier1_degeneracy():
    p = ACParams()
    rng = np.random.default_rng(42)
    actions = list(rng.uniform(0.0, 0.15, size=p.N - 1)) + [1.0]
    base = ExecutionEnv(p, continuous=True)
    br, bp, bi = rollout(base, actions, seed=11)

    degenerates = [
        ("StochasticLiquidity log_std=0", StochasticLiquidityEnv(p, continuous=True, log_std=0.0)),
        ("SqrtImpact power=1", SqrtImpactEnv(p, continuous=True, power=1.0)),
        ("VolumePressure beta=0", VolumePressureEnv(p, continuous=True, beta=0.0)),
    ]
    for name, env in degenerates:
        vr, vp, vi = rollout(env, actions, seed=11)
        check(f"{name} == base (rewards)", np.array_equal(vr, br),
              f"max |dr| = {np.abs(vr-br).max():.3e}")
        check(f"{name} == base (prices)", np.array_equal(vp, bp))


def tier2_closed_forms():
    p = ACParams()
    n_ep = 8000
    tw = twap_schedule(p)

    # sqrt closed form vs MC
    env = SqrtImpactEnv(p, continuous=True, power=0.5)
    sf = evaluate_policy(env, SchedulePolicy(tw), n_episodes=n_ep, seed=100)
    e_theory = env.expected_cost_sqrt(tw)
    z = abs(sf.mean() - e_theory) / (sf.std(ddof=1) / np.sqrt(n_ep))
    check("sqrt: E[IS] closed form vs MC (TWAP)", z < 4,
          f"sim {sf.mean():,.0f} vs theory {e_theory:,.0f} (z={z:.2f})")

    # pressure closed form vs MC
    env = VolumePressureEnv(p, continuous=True)
    sf = evaluate_policy(env, SchedulePolicy(tw), n_episodes=n_ep, seed=200)
    e_theory = env.expected_cost_pressure(tw)
    z = abs(sf.mean() - e_theory) / (sf.std(ddof=1) / np.sqrt(n_ep))
    check("pressure: E[IS] closed form vs MC (TWAP)", z < 4,
          f"sim {sf.mean():,.0f} vs theory {e_theory:,.0f} (z={z:.2f})")

    # liquidity calibration: E[eta_t] = eta  =>  E[IS] equals BASE closed form
    env = StochasticLiquidityEnv(p, continuous=True)
    sf = evaluate_policy(env, SchedulePolicy(tw), n_episodes=2 * n_ep, seed=300)
    base_env = ExecutionEnv(p, continuous=True)
    e_theory = base_env.expected_cost(tw)
    z = abs(sf.mean() - e_theory) / (sf.std(ddof=1) / np.sqrt(2 * n_ep))
    check("liquidity: E[IS](deterministic sched) == base closed form", z < 4,
          f"sim {sf.mean():,.0f} vs base theory {e_theory:,.0f} (z={z:.2f})")


def tier3_mechanisms():
    p = ACParams()

    # liquidity: per-share temp impact is monotone in z
    env = StochasticLiquidityEnv(p, continuous=True)
    env.reset(seed=1)
    n = 20_000.0
    env.z = -1.5
    cheap = env._temp_impact_per_share(n)
    env.z = +1.5
    dear = env._temp_impact_per_share(n)
    check("liquidity: deep (z<0) cheaper than thin (z>0)", cheap < dear,
          f"{cheap:.4f} < {dear:.4f} $/share")

    # sqrt: calibration point at TWAP rate; big trades relatively cheaper
    base = ExecutionEnv(p, continuous=True)
    sq = SqrtImpactEnv(p, continuous=True, power=0.5)
    n_twap = p.X / p.N
    b = base._temp_impact_per_share(n_twap)
    s = sq._temp_impact_per_share(n_twap)
    check("sqrt: per-share impact matches base at TWAP rate",
          abs(b - s) / b < 1e-12, f"{b:.4f} vs {s:.4f}")
    big = 5 * n_twap
    check("sqrt: large trades relatively cheaper than quadratic",
          sq._temp_impact_per_share(big) < base._temp_impact_per_share(big))

    # pressure: bursty schedule costs more than spaced, equal volume
    env = VolumePressureEnv(p, continuous=True)
    spaced = np.full(p.N, p.X / p.N)
    bursty = np.zeros(p.N)
    bursty[::5] = p.X / 10  # all volume in every 5th interval
    check("pressure: bursty > spaced expected cost",
          env.expected_cost_pressure(bursty) > env.expected_cost_pressure(spaced),
          f"{env.expected_cost_pressure(bursty):,.0f} > "
          f"{env.expected_cost_pressure(spaced):,.0f}")

    # pressure: with beta=0 the two coincide with the base closed form
    env0 = VolumePressureEnv(p, continuous=True, beta=0.0)
    b0 = ExecutionEnv(p, continuous=True).expected_cost(spaced)
    check("pressure beta=0: closed form == base closed form",
          abs(env0.expected_cost_pressure(spaced) - b0) / b0 < 1e-12)


if __name__ == "__main__":
    print("--- tier 1: exact degeneracy ---")
    tier1_degeneracy()
    print("--- tier 2: closed forms vs Monte Carlo ---")
    tier2_closed_forms()
    print("--- tier 3: mechanisms ---")
    tier3_mechanisms()
    print("\nAll variant checks passed." if OK else "\nSOME CHECKS FAILED.")
    sys.exit(0 if OK else 1)
