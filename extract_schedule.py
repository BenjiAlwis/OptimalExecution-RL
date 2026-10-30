"""extract_schedule.py — week-4 milestone plot from a ppo_continuous.py checkpoint.

Loads a trained agent.pt (see the torch.save line added at the end of
train()), rolls the GREEDY policy (mean action, no sampling) through a
noise-free copy of the execution environment, and overlays the resulting
holdings trajectory on the analytical Almgren-Chriss sinh curve. Also scores
the greedy schedule with the closed-form J = E[IS] + lam * Var[IS].

Run from the ppo_repro repo root (needs execution_env.py / baselines.py there):

    python extract_schedule.py                       # newest OptimalExecution checkpoint
    python extract_schedule.py --checkpoint runs/<run>/agent.pt
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution_env import ACParams, ExecutionEnv
from baselines import ac_schedule, twap_schedule
from ppo_continuous import Agent

ACTION_CLIP = 8.0  # must match the Box bounds in execution_gym.py (ClipAction range)


class ZeroRng:
    """Noise-free price path for deterministic schedule extraction."""

    def standard_normal(self, *a, **k):
        return 0.0


def sigmoid(u: float) -> float:
    return 1.0 / (1.0 + np.exp(-u))


def greedy_schedule(agent: Agent, p: ACParams) -> np.ndarray:
    env = ExecutionEnv(p, continuous=True)
    env.rng = ZeroRng()
    obs, done = env.reset(), False
    with torch.no_grad():
        while not done:
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            u = float(agent.actor_mean(x).squeeze())
            u = float(np.clip(u, -ACTION_CLIP, ACTION_CLIP))  # mirror ClipAction
            obs, _, done, _ = env.step(sigmoid(u))
    return np.array(env.trades)


def holdings(trades: np.ndarray, X: float) -> np.ndarray:
    return np.concatenate([[X], X - np.cumsum(trades)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to agent.pt; default: newest OptimalExecution run that has one")
    ap.add_argument("--out", default="figures/trajectory_overlay.png")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        cands = sorted(glob.glob("runs/*OptimalExecution*/agent.pt"), key=os.path.getmtime)
        if not cands:
            raise SystemExit(
                "No runs/*OptimalExecution*/agent.pt found. Add the torch.save "
                "line at the end of train() in ppo_continuous.py and retrain, "
                "or pass --checkpoint explicitly."
            )
        ckpt = cands[-1]
    print(f"checkpoint: {ckpt}")

    agent = Agent(obs_dim=3, act_dim=1)
    agent.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agent.eval()

    p = ACParams()
    ref = ExecutionEnv(p, continuous=True)
    ppo = greedy_schedule(agent, p)
    ac = ac_schedule(p)
    tw = twap_schedule(p)

    def J(s):
        return ref.expected_cost(s) + p.lam * ref.cost_variance(s)

    print(f"J(PPO greedy) = {J(ppo):>13,.0f}")
    print(f"J(AC optimal) = {J(ac):>13,.0f}   gap: {100*(J(ppo)-J(ac))/J(ac):+.3f}%")
    print(f"J(TWAP)       = {J(tw):>13,.0f}")

    t = np.arange(p.N + 1) * p.tau
    x_ppo, x_ac, x_tw = (holdings(s, p.X) / p.X for s in (ppo, ac, tw))
    print(f"max |PPO - AC| holdings deviation: {np.abs(x_ppo - x_ac).max()*100:.2f}% of X")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(t, x_ac, color="goldenrod", lw=3, label="AC analytical (sinh)")
    axes[0].plot(t, x_ppo, color="teal", lw=1.8, ls="--", label="PPO greedy (torch)")
    axes[0].plot(t, x_tw, color="gray", lw=1, ls=":", label="TWAP")
    axes[0].set_xlabel("time (days)")
    axes[0].set_ylabel("holdings  $x_t / X$")
    axes[0].set_title("PPO recovers the Almgren-Chriss trajectory")
    axes[0].legend()

    axes[1].bar(np.arange(p.N) - 0.2, ac / p.X * 100, width=0.4,
                color="goldenrod", label="AC analytical")
    axes[1].bar(np.arange(p.N) + 0.2, ppo / p.X * 100, width=0.4,
                color="teal", label="PPO greedy")
    axes[1].set_xlabel("interval $k$")
    axes[1].set_ylabel("trade size  $n_k / X$  (%)")
    axes[1].set_title("Per-interval trades")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
