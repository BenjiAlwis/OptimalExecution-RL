"""checkpoint_trajectory.py — diagnose the OU non-convergence.

Scores EVERY periodic checkpoint (not just late ones) for every seed, so the
question "is this a stationary wander, a late decline, or slow convergence?"
is answered by looking rather than guessing.

Outputs
-------
figures/checkpoint_trajectory_ou.png
    Panel A: greedy gain vs AC (%) against update number, one line per seed,
             with the AC line (0%) and the tilted-AC bound (+7.75%) marked.
    Panel B: policy log-std against update number (if recoverable from the
             checkpoint), the entropy-collapse diagnostic.
    Panel C: distribution of gains in the late half, per seed (box plot) —
             how wide is the wander once "converged"?

Printed table
-------------
    per-seed: early-half mean, late-half mean, late-half std, best, worst,
    plus a trend test (Spearman correlation of gain vs update over the late
    half) distinguishing a stationary wander (rho ~ 0) from a systematic
    late decline (rho < 0).

Interpretation guide (printed at the end):
    stationary wander  -> add an LR schedule / entropy floor; report
                          mean-over-late-checkpoints
    late decline       -> entropy collapse or value drift; the fix is an
                          entropy floor or early stopping on eval
    still improving    -> undertrained; more steps genuinely help

Run from the ppo_repro repo root:
    python checkpoint_trajectory.py
    python checkpoint_trajectory.py --tag ckpt --n-episodes 8000
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution_env import ACParams
from execution_env_variants import StochasticLiquidityEnv
from baselines import ac_schedule, SchedulePolicy, evaluate_policy
from evaluate_variant_policy import load_agent, GreedyPolicy

TILTED_AC_BOUND = 7.751  # % gain over AC achieved by the one-parameter heuristic


def mc_J(env_ctor, policy_ctor, lam, n_ep, seed=9000):
    sf = evaluate_policy(env_ctor(), policy_ctor(), n_episodes=n_ep, seed=seed)
    return sf.mean() + lam * sf.var(ddof=1)


def policy_log_std(ckpt_path):
    """Best-effort extraction of the policy log-std from a state_dict."""
    sd = torch.load(ckpt_path, map_location="cpu")
    for k, v in sd.items():
        if "logstd" in k.replace("_", "").lower():
            return float(np.mean(v.detach().numpy()))
    return np.nan


def spearman(x, y):
    """Spearman rank correlation without scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ckpt", help="run tag to scan (default: ckpt)")
    ap.add_argument("--n-episodes", type=int, default=8000,
                    help="MC episodes per checkpoint (default 8000; SE ~ 7k on J)")
    ap.add_argument("--out", default="figures/checkpoint_trajectory_ou.png")
    args = ap.parse_args()

    p = ACParams()
    lam = p.lam
    mk = lambda: StochasticLiquidityEnv(p, continuous=True)

    J_ac = mc_J(mk, lambda: SchedulePolicy(ac_schedule(p)), lam, args.n_episodes)
    print(f"AC (optimal open-loop) J = {J_ac:,.0f}   "
          f"[tilted-AC bound: +{TILTED_AC_BOUND:.3f}%]")
    print(f"scoring with {args.n_episodes:,} episodes per checkpoint\n")

    pattern = f"runs/*OptimalExecutionOU*__{args.tag}__*/agent_*.pt"
    by_seed = defaultdict(list)
    for ck in glob.glob(pattern):
        seed = int(re.search(r"seed(\d)", ck).group(1))
        upd = int(re.search(r"agent_(\d+)\.pt", ck).group(1))
        by_seed[seed].append((upd, ck))
    if not by_seed:
        raise SystemExit(f"No checkpoints matched {pattern}")

    results = {}
    for seed in sorted(by_seed):
        pts = sorted(by_seed[seed])
        gains, stds, upds = [], [], []
        for upd, ck in pts:
            agent = load_agent(ck, 4)
            J = mc_J(mk, lambda: GreedyPolicy(agent), lam, args.n_episodes)
            gains.append(100 * (J_ac - J) / J_ac)
            stds.append(policy_log_std(ck))
            upds.append(upd)
            print(f"  seed {seed} update {upd:4d}: gain {gains[-1]:+7.3f}%")
        results[seed] = (np.array(upds), np.array(gains), np.array(stds))

    # ------------------------------------------------------------- table
    print("\n" + "=" * 74)
    print(f"{'seed':>4} {'early mean':>11} {'late mean':>10} {'late std':>9} "
          f"{'best':>8} {'worst':>8} {'trend rho':>10}")
    print("-" * 74)
    late_means = []
    for seed, (u, g, _) in results.items():
        half = len(u) // 2
        early, late = g[:half], g[half:]
        rho = spearman(u[half:], late)
        late_means.append(late.mean())
        print(f"{seed:>4} {early.mean():>10.3f}% {late.mean():>9.3f}% "
              f"{late.std():>8.3f} {g.max():>7.2f}% {g.min():>7.2f}% {rho:>10.2f}")
    print("-" * 74)
    print(f"across seeds, late-half mean: {np.mean(late_means):+.3f}% "
          f"(± {np.std(late_means):.3f})")

    all_late_rho = [spearman(u[len(u)//2:], g[len(u)//2:]) for u, g, _ in results.values()]
    mean_rho = np.nanmean(all_late_rho)
    mean_late_std = np.mean([g[len(u)//2:].std() for u, g, _ in results.values()])
    print(f"mean late-half trend rho: {mean_rho:+.2f}   "
          f"mean late-half std: {mean_late_std:.2f} pp")
    print("\ndiagnosis:")
    if mean_rho < -0.5:
        print("  LATE DECLINE — performance systematically degrades late in training.")
        print("  Likely entropy collapse or value drift. Fix: entropy floor, or")
        print("  early stopping on a periodic greedy evaluation.")
    elif mean_rho > 0.5:
        print("  STILL IMPROVING — the runs are budget-limited, not unstable.")
        print("  Fix: more steps (and keep periodic checkpointing to confirm).")
    elif mean_late_std > 3.0:
        print("  STATIONARY WANDER — no trend, but large checkpoint-to-checkpoint")
        print("  variation. The policy is random-walking on a flat, noisy plateau.")
        print("  Fix: anneal the learning rate to ~0 over training (CleanRL-style),")
        print("  and report mean-over-late-checkpoints rather than final weights.")
    else:
        print("  CONVERGED — late-half variation is small; final-checkpoint scores")
        print("  are trustworthy in this setting.")

    # -------------------------------------------------------------- plot
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for seed, (u, g, _) in results.items():
        ax.plot(u, g, marker="o", ms=3, lw=1.5, label=f"seed {seed}")
    ax.axhline(0, color="goldenrod", ls="--", lw=2, label="AC (open-loop optimum)")
    ax.axhline(TILTED_AC_BOUND, color="gray", ls=":", lw=1.5, label="tilted-AC bound")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("greedy gain vs AC (%)")
    ax.set_title("A. Performance across training")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    any_std = False
    for seed, (u, _, s) in results.items():
        if not np.all(np.isnan(s)):
            ax.plot(u, np.exp(s), marker="o", ms=3, lw=1.5, label=f"seed {seed}")
            any_std = True
    ax.set_xlabel("PPO update")
    ax.set_ylabel("policy std  exp(log_std)")
    ax.set_title("B. Exploration std (entropy-collapse check)")
    if any_std:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "log_std not found in state_dict",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    data = [g[len(u)//2:] for u, g, _ in results.values()]
    ax.boxplot(data, labels=[f"s{s}" for s in results])
    ax.axhline(0, color="goldenrod", ls="--", lw=2)
    ax.axhline(TILTED_AC_BOUND, color="gray", ls=":", lw=1.5)
    ax.set_ylabel("greedy gain vs AC (%)")
    ax.set_title("C. Late-half spread per seed")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
