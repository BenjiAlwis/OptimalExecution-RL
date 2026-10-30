"""evaluate_variant_policy.py — score week-5 checkpoints against their targets.

For each runs/*OptimalExecution{OU,Sqrt,Pressure}*/agent.pt:
  - Monte Carlo greedy-policy evaluation (common random numbers) vs. TWAP and
    AC in the same variant env: mean IS, std IS, empirical J.
  - OU additionally: the "opportunism plot" — greedy trade size vs. observed
    liquidity state z at fixed (inventory, time) slices, the behavioral
    signature no open-loop schedule can produce.

Targets from analyze_variants.py:
  OU:       beat AC (MC J = 3,627,452) by more than the tilted-AC bound (7.75%)
  Sqrt:     match numeric open-loop optimum (J = 3,202,121; schedule in
            opt_schedule_sqrt.npy)
  Pressure: match AC (J = 3,287,702)

Run from the ppo_repro repo root:  python evaluate_variant_policy.py
(Written for the torch stack; not executable in the build sandbox — if
Agent's constructor or state_dict keys differ, adjust load_agent.)
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution_env import ACParams, ExecutionEnv
from execution_env_variants import (
    StochasticLiquidityEnv, SqrtImpactEnv, VolumePressureEnv,
)
from baselines import twap_schedule, ac_schedule, SchedulePolicy, evaluate_policy
from ppo_continuous import Agent

N_EP = 20_000
ACTION_BOUND = 8.0

VARIANTS = {
    "OptimalExecutionOU": (StochasticLiquidityEnv, 4),
    "OptimalExecutionSqrt": (SqrtImpactEnv, 3),
    "OptimalExecutionPressure": (VolumePressureEnv, 4),
}


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def load_agent(ckpt, obs_dim):
    agent = Agent(obs_dim=obs_dim, act_dim=1)
    agent.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agent.eval()
    return agent


class GreedyPolicy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        pass

    def __call__(self, obs) -> float:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            u = float(self.agent.actor_mean(x).squeeze())
        return sigmoid(np.clip(u, -ACTION_BOUND, ACTION_BOUND))


def mc_J(env_ctor, policy_ctor, lam, seed=9000):
    sf = evaluate_policy(env_ctor(), policy_ctor(), n_episodes=N_EP, seed=seed)
    return sf.mean(), sf.std(ddof=1), sf.mean() + lam * sf.var(ddof=1)


def opportunism_plot(agent, p, out="figures/opportunism_ou.png"):
    """Greedy action vs liquidity state z at representative (x, t) slices."""
    zs = np.linspace(-2.5, 2.5, 101)
    slices = [(0.9, 0.9), (0.5, 0.5), (0.3, 0.2)]  # (inventory frac, time-left frac)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for inv, tleft in slices:
        obs = np.stack([np.full_like(zs, inv), np.full_like(zs, tleft),
                        np.zeros_like(zs), zs], axis=1).astype(np.float32)
        with torch.no_grad():
            u = agent.actor_mean(torch.tensor(obs)).squeeze(-1).numpy()
        frac = sigmoid(np.clip(u, -ACTION_BOUND, ACTION_BOUND))
        ax.plot(zs, 100 * frac, lw=2, label=f"x/X={inv}, time left={tleft}")
    ax.set_xlabel("liquidity state z  (z<0 = deep/cheap, z>0 = thin/expensive)")
    ax.set_ylabel("greedy sell fraction of remaining (%)")
    ax.set_title("Learned opportunism: trade size responds to liquidity")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = ACParams()
    lam = p.lam
    tw, ac = twap_schedule(p), ac_schedule(p)

    ckpts = sorted(glob.glob("runs/*OptimalExecution*[!0-9]*/agent.pt")) or \
        sorted(glob.glob("runs/*OptimalExecution*/agent.pt"))
    groups = {}
    for c in ckpts:
        for key in VARIANTS:
            if key in c:
                groups.setdefault(key, []).append(c)

    if not groups:
        raise SystemExit("No variant checkpoints found under runs/.")

    for key, (cls, obs_dim) in VARIANTS.items():
        if key not in groups:
            continue
        mk = lambda: cls(p, continuous=True)
        print("\n" + "=" * 70)
        print(f"{key}  ({cls.__name__})")
        m, s, J = mc_J(mk, lambda: SchedulePolicy(tw), lam)
        print(f"{'TWAP':<22} J = {J:>12,.0f}  (mean {m:,.0f}, std {s:,.0f})")
        m, s, J_ac = mc_J(mk, lambda: SchedulePolicy(ac), lam)
        print(f"{'AC':<22} J = {J_ac:>12,.0f}  (mean {m:,.0f}, std {s:,.0f})")
        opt_file = {"OptimalExecutionSqrt": "data/opt_schedule_sqrt.npy",
                    "OptimalExecutionPressure": "data/opt_schedule_pressure.npy"}.get(key)
        if opt_file and os.path.exists(opt_file):
            opt = np.load(opt_file)
            m, s, J_opt = mc_J(mk, lambda: SchedulePolicy(opt), lam)
            print(f"{'numeric open-loop opt':<22} J = {J_opt:>12,.0f}")

        for ckpt in groups[key]:
            seed = re.search(r"seed(\d)", ckpt).group(1)
            agent = load_agent(ckpt, obs_dim)
            m, s, J = mc_J(mk, lambda: GreedyPolicy(agent), lam)
            vs_ac = 100 * (J_ac - J) / J_ac
            print(f"{'PPO greedy seed ' + seed:<22} J = {J:>12,.0f}  "
                  f"(vs AC: {vs_ac:+.3f}%)")

        if key == "OptimalExecutionOU":
            opportunism_plot(load_agent(groups[key][-1], obs_dim), p)


if __name__ == "__main__":
    main()
