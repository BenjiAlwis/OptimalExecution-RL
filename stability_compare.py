"""stability_compare.py — which objective mode gives a STABLE OU policy?

Context: on OptimalExecutionOU-v0 the kl-mode policy does not converge. Late in
training it mostly sits in a good band (+3% to +10% vs AC) but roughly 1 in 5
checkpoints craters (-15% to -33%), fully recovering within 50 updates. The
learning-rate anneal was already enabled in all those runs and the exploration
std is healthy and plateaued, so neither explains it.

Leading hypothesis: the adaptive-KL penalty bounds the AVERAGE divergence per
update, not the per-sample ratio. One unusually aggressive batch can take an
outsized step before beta reacts (logged approx-KL spikes of 0.06-0.07 against
a typical 0.003). Clipping bounds every sample's ratio directly, so it should
not produce the same excursions. Secondary hypothesis: advantage normalisation
inflates updates in calm-liquidity batches (--no-norm-adv arm).

This script scores every periodic checkpoint for each mode/arm and reports the
statistics that actually characterise a skewed distribution: MEDIAN and IQR
(not mean, which rare -30% outliers dominate), plus outlier frequency and the
approx-KL tail from metrics.csv.

Prerequisite: runs with periodic checkpointing (agent_XXXX.pt), e.g.

    for m in kl clip; do for s in 1 2 3 4 5; do
      python ppo_continuous.py --env-id "execution_gym:OptimalExecutionOU-v0" \
        --mode $m --seed $s --gamma 1.0 --no-clip-vloss \
        --total-timesteps 1000000 --tag stab_$m &
      while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
    done; done; wait

Then:  python stability_compare.py --tags stab_kl stab_clip

Add an arm with --no-norm-adv and --tag stab_kl_nonorm to test hypothesis 2.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution_env import ACParams
from execution_env_variants import StochasticLiquidityEnv
from baselines import ac_schedule, SchedulePolicy, evaluate_policy
from evaluate_variant_policy import load_agent, GreedyPolicy

TILTED_AC_BOUND = 7.751
OUTLIER_THRESHOLD = -5.0  # a checkpoint below this much vs AC counts as a blow-up
LATE_FRACTION = 0.5       # analyse the last half of training


def mc_J(env_ctor, policy_ctor, lam, n_ep, seed=9000):
    sf = evaluate_policy(env_ctor(), policy_ctor(), n_episodes=n_ep, seed=seed)
    return sf.mean() + lam * sf.var(ddof=1)


def kl_tail(run_dir):
    """(median, p99, max) of approx_kl for one run, from metrics.csv."""
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return (np.nan,) * 3
    df = pd.read_csv(path)
    kl = df[df["metric"].astype(str).str.contains("kl", case=False, na=False)]
    kl = kl[~kl["metric"].astype(str).str.contains("beta", case=False, na=False)]
    if kl.empty:
        return (np.nan,) * 3
    v = kl["value"].values
    return float(np.median(v)), float(np.percentile(v, 99)), float(v.max())


def collect(tag, n_ep, lam, mk, J_ac):
    """Score all late checkpoints for one tag. Returns per-seed gain arrays."""
    by_seed = defaultdict(list)
    for ck in glob.glob(f"runs/*OptimalExecutionOU*__{tag}__*/agent_*.pt"):
        seed = int(re.search(r"seed(\d)", ck).group(1))
        upd = int(re.search(r"agent_(\d+)\.pt", ck).group(1))
        by_seed[seed].append((upd, ck))

    out = {}
    for seed in sorted(by_seed):
        pts = sorted(by_seed[seed])
        cutoff = pts[int(len(pts) * (1 - LATE_FRACTION))][0]
        gains = []
        for upd, ck in pts:
            if upd < cutoff:
                continue
            J = mc_J(mk, lambda: GreedyPolicy(load_agent(ck, 4)), lam, n_ep)
            gains.append(100 * (J_ac - J) / J_ac)
        out[seed] = np.array(gains)
        if gains:
            print(f"    seed {seed}: {len(gains)} late ckpts, "
                  f"median {np.median(gains):+.2f}%, min {min(gains):+.2f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["stab_kl", "stab_clip"],
                    help="run tags to compare (one per arm)")
    ap.add_argument("--n-episodes", type=int, default=8000)
    ap.add_argument("--out", default="figures/stability_compare_ou.png")
    args = ap.parse_args()

    p = ACParams()
    lam = p.lam
    mk = lambda: StochasticLiquidityEnv(p, continuous=True)
    J_ac = mc_J(mk, lambda: SchedulePolicy(ac_schedule(p)), lam, args.n_episodes)
    print(f"AC (optimal open-loop) J = {J_ac:,.0f}   "
          f"[tilted-AC bound +{TILTED_AC_BOUND:.2f}%]\n")

    arms = {}
    for tag in args.tags:
        print(f"--- {tag} ---")
        arms[tag] = collect(tag, args.n_episodes, lam, mk, J_ac)
        if not arms[tag]:
            print(f"    (no checkpoints matched tag {tag})")

    # ------------------------------------------------------------- table
    print("\n" + "=" * 88)
    print(f"{'arm':<14} {'ckpts':>6} {'median':>9} {'IQR':>8} {'p10':>9} "
          f"{'worst':>9} {'blowups':>9} {'KL p99':>9}")
    print("-" * 88)
    summary = {}
    for tag, per_seed in arms.items():
        if not per_seed:
            continue
        allg = np.concatenate(list(per_seed.values()))
        q1, q3 = np.percentile(allg, [25, 75])
        blow = 100 * np.mean(allg < OUTLIER_THRESHOLD)
        kls = [kl_tail(d)[1] for d in
               glob.glob(f"runs/*OptimalExecutionOU*__{tag}__*") if os.path.isdir(d)]
        kl99 = np.nanmean(kls) if kls else np.nan
        summary[tag] = (np.median(allg), q3 - q1, blow)
        print(f"{tag:<14} {len(allg):>6} {np.median(allg):>8.2f}% {q3-q1:>7.2f} "
              f"{np.percentile(allg,10):>8.2f}% {allg.min():>8.2f}% "
              f"{blow:>8.1f}% {kl99:>9.4f}")
    print("-" * 88)
    print("median/IQR are the right summary here: the distribution is left-skewed,")
    print(f"so the mean is dominated by rare blow-ups (< {OUTLIER_THRESHOLD:.0f}% vs AC).")

    if len(summary) >= 2:
        tags = list(summary)
        best_stable = min(tags, key=lambda t: (summary[t][2], summary[t][1]))
        print(f"\nmost stable arm: {best_stable} "
              f"(blow-ups {summary[best_stable][2]:.1f}%, IQR {summary[best_stable][1]:.2f}pp)")
        others = [t for t in tags if t != best_stable]
        if all(summary[best_stable][2] + 5 < summary[t][2] for t in others):
            print("  -> clear stability difference: the objective/normalisation choice,")
            print("     not step size, drives the excursions.")
        else:
            print("  -> no clear stability difference between arms; the instability is")
            print("     not explained by this factor. Next suspects: reward-scale tails,")
            print("     or the OU state's effect on advantage estimation.")

    # -------------------------------------------------------------- plot
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = len([t for t in arms if arms[t]])
    fig, axes = plt.subplots(1, max(n, 1) + 1, figsize=(5 * (max(n, 1) + 1), 4.5))
    if n + 1 == 1:
        axes = [axes]

    for ax, (tag, per_seed) in zip(axes, [(t, arms[t]) for t in arms if arms[t]]):
        ax.boxplot([per_seed[s] for s in sorted(per_seed)],
                   labels=[f"s{s}" for s in sorted(per_seed)])
        ax.axhline(0, color="goldenrod", ls="--", lw=2, label="AC")
        ax.axhline(TILTED_AC_BOUND, color="gray", ls=":", lw=1.5, label="tilted-AC")
        ax.set_title(f"{tag}: late-half spread")
        ax.set_ylabel("greedy gain vs AC (%)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)

    ax = axes[-1]
    for tag, per_seed in arms.items():
        if not per_seed:
            continue
        allg = np.sort(np.concatenate(list(per_seed.values())))
        ax.plot(allg, np.linspace(0, 1, len(allg)), lw=2, marker="o", ms=3, label=tag)
    ax.axvline(0, color="goldenrod", ls="--", lw=2)
    ax.axvline(TILTED_AC_BOUND, color="gray", ls=":", lw=1.5)
    ax.set_xlabel("greedy gain vs AC (%)")
    ax.set_ylabel("cumulative fraction of checkpoints")
    ax.set_title("CDF: left tail is the whole story")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
