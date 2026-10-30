"""test_calibrated_env.py — validate CalibratedExecutionEnv before it is used.

Same discipline as weeks 5-6: a new environment is not trusted until it
degenerates EXACTLY to an already-validated one, and until its calibration
reproduces the target it claims to match.

Checks
------
1. power=1, log_std=0 is bit-identical to the base ExecutionEnv (same seed,
   same actions, same rewards and prices).
2. power=1, log_std>0 is bit-identical to week-5 StochasticLiquidityEnv.
3. power<1, log_std=0 is bit-identical to week-5 SqrtImpactEnv (with the
   matching coefficient).
4. Square-root-law calibration: a TWAP schedule in the calibrated env costs
   what the law says, at any exponent.
5. E[c_t] = c under the OU process (lognormal mean correction is right).
6. expected_cost_power matches Monte Carlo, with and without OU liquidity.
7. Builder: kappa*T comes out at the requested target.

Run: python test_calibrated_env.py
"""

from __future__ import annotations

import sys

import numpy as np

from execution_env import ACParams, ExecutionEnv
from execution_env_variants import StochasticLiquidityEnv, SqrtImpactEnv
from calibrated_env import (
    CalibratedExecutionEnv, sqrt_law_cost_per_share, impact_coef_for_power,
    build_from_calibration,
)
from baselines import twap_schedule, SchedulePolicy, evaluate_policy

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def rollout(env, actions, seed):
    env.reset(seed=seed)
    r, s = [], []
    for a in actions:
        _, rew, done, _ = env.step(a)
        r.append(rew)
        s.append(env.S)
    return np.array(r), np.array(s)


def test_degeneracy():
    p = ACParams()
    rng = np.random.default_rng(7)
    acts = list(rng.uniform(0.0, 0.12, size=p.N - 1)) + [1.0]

    base = ExecutionEnv(p, continuous=True)
    cal = CalibratedExecutionEnv(p, power=1.0, log_std=0.0, continuous=True)
    rb, sb = rollout(base, acts, 21)
    rc, sc = rollout(cal, acts, 21)
    check("power=1, log_std=0 == base ExecutionEnv",
          np.array_equal(rb, rc) and np.array_equal(sb, sc),
          f"max |dr| = {np.abs(rb - rc).max():.3e}")

    ou = StochasticLiquidityEnv(p, continuous=True, theta=12.0, log_std=0.4)
    cal_ou = CalibratedExecutionEnv(p, power=1.0, theta=12.0, log_std=0.4,
                                    continuous=True)
    ro, so = rollout(ou, acts, 33)
    rc, sc = rollout(cal_ou, acts, 33)
    check("power=1, log_std>0 == StochasticLiquidityEnv",
          np.allclose(ro, rc, rtol=0, atol=1e-12) and np.allclose(so, sc, atol=1e-12),
          f"max |dr| = {np.abs(ro - rc).max():.3e}")

    sq = SqrtImpactEnv(p, continuous=True, power=0.5)
    cal_sq = CalibratedExecutionEnv(p, power=0.5, impact_coef=sq.c,
                                    continuous=True)
    rs, ss = rollout(sq, acts, 44)
    rc, sc = rollout(cal_sq, acts, 44)
    check("power<1, log_std=0 == SqrtImpactEnv",
          np.allclose(rs, rc, atol=1e-12) and np.allclose(ss, sc, atol=1e-12),
          f"max |dr| = {np.abs(rs - rc).max():.3e}")


def test_law_calibration():
    """A TWAP schedule must cost what the square-root law says, any exponent."""
    X, adv, sigma, S0 = 100_000.0, 10_000_000.0, 0.37, 27.0
    target = sqrt_law_cost_per_share(X, adv, sigma, S0)
    for power in (1.0, 0.8, 0.5):
        c = impact_coef_for_power(X, adv, sigma, S0, power)
        # cost per share at the TWAP rate X/T
        got = c * (X / 1.0) ** power
        check(f"law calibration at power={power}", abs(got - target) / target < 1e-12,
              f"{got:.6g} vs {target:.6g} $/share")


def test_episode_streams_differ():
    """Liquidity must differ across episodes even with an explicit state_seed.

    Deriving the stream from state_seed alone made every episode replay one
    identical liquidity path — invisible in any single-episode check, and
    ruinous for training, since the agent could memorise the path instead of
    learning to react to the state.
    """
    p = ACParams()
    env = CalibratedExecutionEnv(p, power=1.0, theta=30.0, log_std=0.45,
                                 continuous=True, state_seed=5)
    zs = []
    for ep in range(8):
        env.reset(seed=ep)
        zs.append(env.z)
    check("episodes get distinct liquidity paths", len(set(np.round(zs, 9))) == 8,
          f"{len(set(np.round(zs, 9)))}/8 distinct initial z")
    # and still reproducible for a given seed
    env.reset(seed=3)
    z3 = env.z
    env.reset(seed=3)
    check("same seed reproduces the same path", env.z == z3)


def test_ou_mean():
    """E[c_t] must equal c, or the law calibration is silently off."""
    p = ACParams()
    env = CalibratedExecutionEnv(p, power=1.0, impact_coef=p.eta,
                                 theta=30.0, log_std=0.45, continuous=True,
                                 state_seed=5)
    vals = []
    for ep in range(400):
        env.reset(seed=ep)
        for _ in range(p.N - 1):
            vals.append(env.coef_now)
            env.step(0.01)
    m = float(np.mean(vals))
    check("E[c_t] == c under OU", abs(m - p.eta) / p.eta < 0.03,
          f"mean {m:.6g} vs c {p.eta:.6g} (rel {abs(m - p.eta) / p.eta:.3%})")


def test_closed_form():
    p = ACParams()
    tw = twap_schedule(p)
    for label, kw in (("no OU", dict(power=0.6)),
                      ("with OU", dict(power=0.6, theta=35.0, log_std=0.4))):
        c = impact_coef_for_power(p.X, 5e7, p.sigma, p.S0, 0.6)
        env = CalibratedExecutionEnv(p, impact_coef=c, continuous=True, **kw)
        theory = env.expected_cost_power(tw)
        sf = evaluate_policy(CalibratedExecutionEnv(p, impact_coef=c,
                                                    continuous=True, **kw),
                             SchedulePolicy(tw), n_episodes=6000, seed=99)
        z = abs(sf.mean() - theory) / (sf.std(ddof=1) / np.sqrt(6000))
        check(f"expected_cost_power vs MC ({label})", z < 4,
              f"sim {sf.mean():,.0f} vs theory {theory:,.0f} (z={z:.2f})")


def test_builder(tmp_csv="_test_cal.csv"):
    import pandas as pd
    pd.DataFrame([
        dict(ticker="AAA", mid_px=27.0, sigma=0.37, eps=0.005, gamma=5e-7,
             adv=1e7, power=0.81, theta=32.0, ou_log_std=0.39),
        dict(ticker="BBB", mid_px=570.0, sigma=5.5, eps=0.137, gamma=1.2e-4,
             adv=7e5, power=0.55, theta=63.0, ou_log_std=0.41),
    ]).to_csv(tmp_csv, index=False)

    for target in (1.0, 2.0, 4.0):
        cfgs = build_from_calibration(tmp_csv, pct_adv=1.0,
                                      target_kappa_T=target)
        ok = all(abs(c.kappa_T - target) / target < 0.05 for c in cfgs.values())
        check(f"builder hits kappa*T = {target}", ok,
              ", ".join(f"{k}={v.kappa_T:.2f}" for k, v in cfgs.items()))

    cfgs = build_from_calibration(tmp_csv, pct_adv=1.0, lam=2e-6)
    for c in cfgs.values():
        # the environment's own impact coefficient must reproduce the law
        target = sqrt_law_cost_per_share(c.X, {"AAA": 1e7, "BBB": 7e5}[c.ticker],
                                         c.params.sigma, c.params.S0)
        got = c.impact_coef * c.X ** c.power
        check(f"{c.ticker}: env impact matches the law",
              abs(got - target) / target < 1e-10,
              f"{got:.6g} vs {target:.6g} $/share at power {c.power}")
    import os
    os.remove(tmp_csv)


def main():
    print("--- degeneracy to validated environments ---")
    test_degeneracy()
    print("--- square-root-law calibration ---")
    test_law_calibration()
    print("--- episode RNG streams ---")
    test_episode_streams_differ()
    print("--- OU mean correction ---")
    test_ou_mean()
    print("--- closed form vs Monte Carlo ---")
    test_closed_form()
    print("--- builder ---")
    test_builder()
    print("\nAll calibrated-env checks passed." if OK else "\nSOME CHECKS FAILED.")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
