"""Week-5 baseline analysis for the three AC-breaking variants.

Builds the baseline ladder each variant needs before RL results mean anything:

  TWAP                     naive
  AC (base params)         what a desk running the formula would do
  numeric open-loop opt    best deterministic schedule for the TRUE dynamics
                           (sqrt & pressure: solved as a convex/smooth program;
                            OU: provably equal to AC, no solve needed)
  tilted-AC(alpha)         OU only — one-parameter adaptive heuristic
                           frac_k = ac_frac_k * exp(-alpha * z_k); a cheap
                           LOWER bound on the value of adaptivity that RL
                           must beat to justify itself.

All Monte Carlo comparisons use common random numbers (same seeds across
policies) so differences are policy differences, not noise.

Deterministic-schedule objectives use exact closed forms:
  E[IS]   variant-specific (expected_cost / expected_cost_sqrt / _pressure)
  Var[IS] = sigma^2 tau sum x_k^2   (unchanged: price-noise structure is
            identical in all variants)

Run: python analyze_variants.py     (~2-4 min)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from execution_env import ACParams, ExecutionEnv
from execution_env_variants import (
    StochasticLiquidityEnv, SqrtImpactEnv, VolumePressureEnv,
)
from baselines import twap_schedule, ac_schedule, SchedulePolicy, evaluate_policy

N_EP = 10_000
SEED = 5000


def empirical_J(env_ctor, policy_ctor, lam, n_ep=N_EP, seed=SEED):
    env = env_ctor()
    sf = evaluate_policy(env, policy_ctor(), n_episodes=n_ep, seed=seed)
    return sf.mean(), sf.std(ddof=1), sf.mean() + lam * sf.var(ddof=1)


def closed_J(env, schedule, lam, kind):
    fn = {"base": "expected_cost",
          "sqrt": "expected_cost_sqrt",
          "pressure": "expected_cost_pressure"}[kind]
    return getattr(env, fn)(schedule) + lam * env.cost_variance(schedule)


def numeric_open_loop(env, lam, kind, n0):
    """Best deterministic schedule for the variant's true expected dynamics."""
    p = env.p
    N = p.N

    def obj(n):
        return closed_J(env, n, lam, kind) / 1e6  # scale for the optimizer

    res = minimize(
        obj, n0, method="SLSQP",
        bounds=[(0.0, p.X)] * N,
        constraints=[{"type": "eq", "fun": lambda n: n.sum() / p.X - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x, res.success


class TiltedACPolicy:
    """OU-variant heuristic: AC fraction-of-remaining, tilted by liquidity.

    frac_k = clip(ac_frac_k * exp(-alpha * z_k), 0, 1); z is obs[3]. alpha=0
    reproduces AC exactly. Positive alpha trades more when liquidity is deep
    (z<0) and less when thin (z>0).
    """

    def __init__(self, p: ACParams, alpha: float):
        s = ac_schedule(p)
        x = p.X - np.concatenate([[0.0], np.cumsum(s)[:-1]])
        self.base_frac = s / x
        self.alpha = alpha
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, obs) -> float:
        z = obs[3]
        f = self.base_frac[self.k] * np.exp(-self.alpha * z)
        self.k += 1
        return float(np.clip(f, 0.0, 1.0))


def main():
    p = ACParams()
    lam = p.lam
    tw, ac = twap_schedule(p), ac_schedule(p)

    print("=" * 72)
    print("references (base env closed form): "
          f"J_AC = {ExecutionEnv(p).expected_cost(ac) + lam*ExecutionEnv(p).cost_variance(ac):,.0f}   "
          f"J_TWAP = {ExecutionEnv(p).expected_cost(tw) + lam*ExecutionEnv(p).cost_variance(tw):,.0f}")

    # ------------------------------------------------------------- sqrt
    print("\n--- SqrtImpactEnv (P1 broken: sub-quadratic cost) " + "-" * 22)
    env = SqrtImpactEnv(p, continuous=True)
    for name, s in (("TWAP", tw), ("AC (mis-specified)", ac)):
        print(f"{name:<24} closed-form J = {closed_J(env, s, lam, 'sqrt'):>12,.0f}")
    opt, ok = numeric_open_loop(env, lam, "sqrt", n0=ac.copy())
    J_opt = closed_J(env, opt, lam, "sqrt")
    print(f"{'numeric open-loop opt':<24} closed-form J = {J_opt:>12,.0f}   "
          f"(solver ok={ok})")
    print(f"  AC excess over true optimum: "
          f"{100*(closed_J(env, ac, lam, 'sqrt')-J_opt)/J_opt:+.3f}%   "
          f"<- RL's target is to MATCH the numeric optimum")
    np.save("data/opt_schedule_sqrt.npy", opt)

    # --------------------------------------------------------- pressure
    print("\n--- VolumePressureEnv (P3 broken: non-Markov spread) " + "-" * 19)
    env = VolumePressureEnv(p, continuous=True)
    for name, s in (("TWAP", tw), ("AC (mis-specified)", ac)):
        print(f"{name:<24} closed-form J = {closed_J(env, s, lam, 'pressure'):>12,.0f}")
    opt, ok = numeric_open_loop(env, lam, "pressure", n0=tw.copy())
    J_opt = closed_J(env, opt, lam, "pressure")
    print(f"{'numeric open-loop opt':<24} closed-form J = {J_opt:>12,.0f}   "
          f"(solver ok={ok})")
    print(f"  AC excess over true optimum: "
          f"{100*(closed_J(env, ac, lam, 'pressure')-J_opt)/J_opt:+.3f}%   "
          f"<- RL's target is to MATCH the numeric optimum")
    np.save("data/opt_schedule_pressure.npy", opt)

    # --------------------------------------------------------- liquidity
    print("\n--- StochasticLiquidityEnv (P2 broken: adaptivity has value) " + "-" * 11)
    print("Open-loop fact: E[eta_t] = eta  =>  every deterministic schedule has")
    print("the SAME expected cost as in the base env; AC IS the optimal open-loop")
    print("schedule here. Any improvement below is pure adaptivity value.")
    mk_env = lambda: StochasticLiquidityEnv(p, continuous=True)
    m, s, J = empirical_J(mk_env, lambda: SchedulePolicy(tw), lam)
    print(f"{'TWAP':<24} MC J = {J:>12,.0f}   (mean {m:,.0f}, std {s:,.0f})")
    m, s, J_ac_mc = empirical_J(mk_env, lambda: SchedulePolicy(ac), lam)
    print(f"{'AC = open-loop opt':<24} MC J = {J_ac_mc:>12,.0f}   (mean {m:,.0f}, std {s:,.0f})")
    best = (None, np.inf)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        m, s, J = empirical_J(mk_env, lambda a=alpha: TiltedACPolicy(p, a), lam)
        tag = ""
        if J < best[1]:
            best = (alpha, J)
            tag = "  <- best so far"
        print(f"{'tilted-AC alpha=%.2f' % alpha:<24} MC J = {J:>12,.0f}{tag}")
    gain = 100 * (J_ac_mc - best[1]) / J_ac_mc
    print(f"\n  adaptivity lower bound: tilted-AC(alpha={best[0]}) beats the optimal")
    print(f"  open-loop schedule by {gain:.3f}% of J -- RL should beat this bound.")


if __name__ == "__main__":
    main()
