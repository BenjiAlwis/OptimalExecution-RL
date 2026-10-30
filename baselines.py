"""Baseline execution schedules: TWAP and the analytical Almgren-Chriss solution.

Both are expressed as *trade schedules* (shares per interval, deterministic) and
as *policies* (callables mapping env state to the env's expected action format),
so they can be evaluated either in closed form or by rollout in ExecutionEnv.
"""

from __future__ import annotations

import numpy as np

from execution_env import ACParams


# --------------------------------------------------------------- schedules
def twap_schedule(p: ACParams) -> np.ndarray:
    """Equal shares every interval."""
    return np.full(p.N, p.X / p.N)


def ac_schedule(p: ACParams) -> np.ndarray:
    """Almgren-Chriss optimal trade list.

    Holdings trajectory: x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T),
    t_j = j * tau, j = 0..N.  Trade list n_j = x_{j-1} - x_j.
    As lam -> 0, kappa -> 0 and this degenerates to TWAP.
    """
    if p.lam <= 0:
        return twap_schedule(p)
    kappa = p.kappa
    j = np.arange(p.N + 1)
    x = p.X * np.sinh(kappa * (p.T - j * p.tau)) / np.sinh(kappa * p.T)
    return -np.diff(x)


# ---------------------------------------------------------------- policies
class SchedulePolicy:
    """Wraps a deterministic trade schedule as an env policy.

    Emits *continuous* actions (fraction of remaining inventory), which is
    exact; use env(continuous=True) when rolling these out.
    """

    def __init__(self, schedule: np.ndarray):
        self.schedule = np.asarray(schedule, dtype=float)
        self.k = 0
        self.remaining = self.schedule.sum()

    def reset(self):
        self.k = 0
        self.remaining = self.schedule.sum()

    def __call__(self, obs) -> float:
        n_k = self.schedule[self.k]
        frac = 0.0 if self.remaining <= 0 else n_k / self.remaining
        self.remaining -= n_k
        self.k += 1
        return float(np.clip(frac, 0.0, 1.0))


def evaluate_policy(env, policy, n_episodes: int = 1000, seed: int = 0):
    """Roll a policy out and return realized shortfalls (dollars)."""
    shortfalls = np.empty(n_episodes)
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        while not done:
            obs, _, done, info = env.step(policy(obs))
        shortfalls[ep] = info["shortfall"]
    return shortfalls
