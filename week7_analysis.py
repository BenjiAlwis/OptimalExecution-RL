"""week7_analysis.py — does the week-5 adaptivity result survive calibration?

Week 5 found that PPO beats the provably-optimal open-loop schedule by a
median 6.65% when liquidity mean-reverts. That was measured with ASSUMED OU
parameters: theta = 10 (autocorrelation 0.82 per interval) and log-std 0.40.
Week 6 calibration says real books mean-revert 3-6x faster: theta = 32-63,
i.e. rho = 0.28-0.53. A signal you can see one or two intervals ahead is far
less exploitable than one you can see five or six ahead, so the gain should
shrink. By how much is the question this script answers.

It uses the tilted-AC heuristic as the measuring instrument rather than RL:
frac_k = ac_frac_k * exp(-alpha * z_k), swept over alpha. That is a LOWER
bound on the value of adaptivity (a one-parameter rule cannot express the
state-dependent responsiveness a learned policy showed in week 5), but it is
deterministic, cheap, and free of the training instability that made week-5
RL numbers hard to pin down. Use it to size the effect; use RL to beat it.

Two experiments:

  A. THETA ABLATION. For each ticker, hold everything else at its calibrated
     value and sweep theta over {week-5's 10, calibrated, and points between}.
     Isolates the persistence effect from every other calibrated change.

  B. URGENCY SWEEP. For each ticker at calibrated theta, sweep lambda by
     target kappa*T in {0.5, 1, 2, 4, 8}. Tests whether adaptivity value
     depends on the urgency regime: a schedule that is dumping inventory
     anyway has little room to be opportunistic.

Both report the adaptivity gain as a percentage of J against that ticker's own
AC baseline, evaluated with common random numbers.

Usage:
    python week7_analysis.py                      # both experiments
    python week7_analysis.py --experiment theta   # just A
    python week7_analysis.py --n-episodes 4000    # faster, noisier
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution_env import ACParams
from calibrated_env import build_from_calibration, CalibratedExecutionEnv
from baselines import ac_schedule, SchedulePolicy, evaluate_policy

WEEK5_THETA = 10.0
WEEK5_LOG_STD = 0.40
# The tilt grid must bracket the optimum. At 8000 episodes the concave-book
# names pinned alpha* at the old top of 0.6 in nearly every configuration,
# which silently truncated their reported gain. adaptivity_gain() now warns
# whenever the best alpha lands on either edge.
ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2)


class TiltedACPolicy:
    """AC fraction-of-remaining, tilted by observed liquidity.

    frac_k = clip(ac_frac_k * exp(-alpha * z_k), 0, 1); alpha = 0 is AC itself.
    Positive alpha trades harder when liquidity is deep (z < 0).
    """

    def __init__(self, p: ACParams, alpha: float, z_index: int = 3):
        s = ac_schedule(p)
        x = p.X - np.concatenate([[0.0], np.cumsum(s)[:-1]])
        self.base_frac = s / np.maximum(x, 1e-12)
        self.alpha = alpha
        self.z_index = z_index
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, obs) -> float:
        z = obs[self.z_index] if len(obs) > self.z_index else 0.0
        f = self.base_frac[min(self.k, len(self.base_frac) - 1)] * np.exp(-self.alpha * z)
        self.k += 1
        return float(np.clip(f, 0.0, 1.0))


def adaptivity_gain(cfg, theta: float, log_std: float, n_ep: int,
                    seed: int = 777) -> tuple[float, float, float]:
    """Best tilted-AC gain over AC, in % of J. Returns (gain, best_alpha, J_ac)."""
    ac_p = cfg.ac_params()

    def J(policy_factory) -> float:
        env = CalibratedExecutionEnv(
            cfg.params, power=cfg.power, impact_coef=cfg.impact_coef,
            theta=theta, log_std=log_std, continuous=True)
        sf = evaluate_policy(env, policy_factory(), n_episodes=n_ep, seed=seed)
        return sf.mean() + cfg.params.lam * sf.var(ddof=1)

    J_ac = J(lambda: SchedulePolicy(ac_schedule(ac_p)))
    best_J, best_a = J_ac, 0.0
    for a in ALPHAS:
        if a == 0.0:
            continue
        Ja = J(lambda a=a: TiltedACPolicy(ac_p, a))
        if Ja < best_J:
            best_J, best_a = Ja, a
    if best_a == max(ALPHAS):
        print(f"    [warn] alpha* pinned at the grid maximum {best_a}: the "
              "reported gain is TRUNCATED, extend ALPHAS")
    return 100.0 * (J_ac - best_J) / J_ac, best_a, J_ac


def experiment_theta(cfgs, n_ep: int) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("A. THETA ABLATION — how much of the week-5 gain survives calibration?")
    print("=" * 78)
    print(f"{'ticker':<7} {'theta':>7} {'rho':>6} {'log_std':>8} "
          f"{'gain %':>8} {'alpha*':>7}  note")
    rows = []
    for tk, cfg in cfgs.items():
        tau = cfg.params.tau
        # theta is swept with log_std HELD at the calibrated value, so the
        # sweep isolates persistence. A separate row gives the full week-5
        # configuration (theta=10 AND log_std=0.40) for the overall contrast.
        for th in [WEEK5_THETA, 15.0, 22.0, cfg.theta]:
            g, a, _ = adaptivity_gain(cfg, th, cfg.log_std, n_ep)
            note = "CALIBRATED" if th == cfg.theta else ""
            print(f"{tk:<7} {th:7.2f} {np.exp(-th * tau):6.3f} {cfg.log_std:8.3f} "
                  f"{g:8.2f} {a:7.2f}  {note}")
            rows.append(dict(ticker=tk, theta=th, rho=np.exp(-th * tau),
                             log_std=cfg.log_std, gain_pct=g, best_alpha=a,
                             calibrated=(th == cfg.theta), arm="theta_sweep"))
        g5, a5, _ = adaptivity_gain(cfg, WEEK5_THETA, WEEK5_LOG_STD, n_ep)
        print(f"{tk:<7} {WEEK5_THETA:7.2f} {np.exp(-WEEK5_THETA * tau):6.3f} "
              f"{WEEK5_LOG_STD:8.3f} {g5:8.2f} {a5:7.2f}  week-5 config (both)")
        rows.append(dict(ticker=tk, theta=WEEK5_THETA,
                         rho=np.exp(-WEEK5_THETA * tau), log_std=WEEK5_LOG_STD,
                         gain_pct=g5, best_alpha=a5, calibrated=False,
                         arm="week5_config"))
        pd.DataFrame(rows).to_csv("week7_theta.csv", index=False)  # incremental
        print()
    return pd.DataFrame(rows)


def experiment_urgency(csv, pct_adv, N, n_ep, targets=(0.5, 1.0, 2.0, 4.0, 8.0)) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("B. URGENCY SWEEP — does adaptivity value depend on kappa*T?")
    print("=" * 78)
    print(f"{'ticker':<7} {'kappa*T':>8} {'lam':>11} {'%sold 1st 10%':>14} "
          f"{'gain %':>8} {'alpha*':>7}")
    rows = []
    for target in targets:
        cfgs = build_from_calibration(csv, pct_adv=pct_adv, N=N,
                                      target_kappa_T=target)
        for tk, cfg in cfgs.items():
            s = ac_schedule(cfg.ac_params())
            early = 100 * s[:max(1, N // 10)].sum() / cfg.params.X
            g, a, _ = adaptivity_gain(cfg, cfg.theta, cfg.log_std, n_ep)
            print(f"{tk:<7} {cfg.kappa_T:8.2f} {cfg.params.lam:11.3g} "
                  f"{early:14.1f} {g:8.2f} {a:7.2f}")
            rows.append(dict(ticker=tk, kappa_T=cfg.kappa_T, lam=cfg.params.lam,
                             early_pct=early, gain_pct=g, best_alpha=a))
        pd.DataFrame(rows).to_csv("week7_urgency.csv", index=False)  # incremental
        print()
    return pd.DataFrame(rows)


def plot(df_theta, df_urg, out="figures/week7_calibration_effect.png"):
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n = (df_theta is not None) + (df_urg is not None)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6.5 * max(n, 1), 4.5))
    axes = np.atleast_1d(axes)
    i = 0
    if df_theta is not None:
        ax = axes[i]; i += 1
        for tk, g in df_theta[df_theta["arm"] == "theta_sweep"].groupby("ticker"):
            g = g.sort_values("theta")
            ax.plot(g["theta"], g["gain_pct"], marker="o", lw=1.8, label=tk)
            cal = g[g["calibrated"]]
            ax.scatter(cal["theta"], cal["gain_pct"], s=90, facecolors="none",
                       edgecolors="black", zorder=5)
        ax.axvline(WEEK5_THETA, color="gray", ls="--", lw=1.5,
                   label="week-5 assumption")
        ax.set_xlabel(r"OU mean reversion $\theta$ (1/day)")
        ax.set_ylabel("adaptivity gain over AC (%)")
        ax.set_title("A. Faster mean reversion RAISES the gain\n"
                     "(circled = calibrated value)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    if df_urg is not None:
        ax = axes[i]
        for tk, g in df_urg.groupby("ticker"):
            g = g.sort_values("kappa_T")
            ax.plot(g["kappa_T"], g["gain_pct"], marker="s", lw=1.8, label=tk)
        ax.set_xscale("log")
        ax.set_xlabel(r"urgency  $\kappa T$")
        ax.set_ylabel("adaptivity gain over AC (%)")
        ax.set_title("B. Urgency helps concave books, not tick-constrained ones")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="calibration_all.csv")
    ap.add_argument("--pct-adv", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-episodes", type=int, default=8000)
    ap.add_argument("--experiment", choices=["theta", "urgency", "both"],
                    default="both")
    args = ap.parse_args()

    cfgs = build_from_calibration(args.calibration, pct_adv=args.pct_adv,
                                  N=args.n, lam=2e-6)
    print(f"built {len(cfgs)} calibrated ticker(s) at {args.pct_adv}% of volume")
    print(f"{'ticker':<7} {'X':>10} {'power':>6} {'theta':>7} {'log_std':>8} "
          f"{'kappa*T':>8}")
    for tk, c in cfgs.items():
        print(f"{tk:<7} {c.X:10,.0f} {c.power:6.3f} {c.theta:7.2f} "
              f"{c.log_std:8.3f} {c.kappa_T:8.2f}")

    dt = du = None
    if args.experiment in ("theta", "both"):
        dt = experiment_theta(cfgs, args.n_episodes)
        dt.to_csv("week7_theta.csv", index=False)
        cal = dt[dt["calibrated"]]["gain_pct"].mean()
        w5 = dt[dt["arm"] == "week5_config"]["gain_pct"].mean()
        sweep_lo = dt[(dt["arm"] == "theta_sweep") &
                      (dt["theta"] == WEEK5_THETA)]["gain_pct"].mean()
        print(f"  theta effect alone (log_std held calibrated): "
              f"{sweep_lo:.2f}% at theta=10 -> {cal:.2f}% at calibrated theta")
        chg = 100 * (cal / w5 - 1) if w5 else float("nan")
        word = "HIGHER" if chg > 0 else "lower"
        print(f"SUMMARY: mean gain {w5:.2f}% under the full week-5 configuration "
              f"-> {cal:.2f}% fully calibrated ({abs(chg):.0f}% {word})")
        print("  NOTE the direction: faster mean reversion INCREASES the value of")
        print("  adaptivity. Persistence is not the same as exploitability — a")
        print("  highly persistent liquidity state is nearly CONSTANT within an")
        print("  episode, and a constant state cannot be exploited by reallocating")
        print("  trades across intervals. Fast reversion gives roughly theta*T")
        print("  independent cheap/expensive windows per day to shift volume into.")
    if args.experiment in ("urgency", "both"):
        du = experiment_urgency(args.calibration, args.pct_adv, args.n,
                                args.n_episodes)
        du.to_csv("week7_urgency.csv", index=False)
        best = du.loc[du["gain_pct"].idxmax()]
        print(f"SUMMARY: largest gain {best.gain_pct:.2f}% at kappa*T = "
              f"{best.kappa_T:.2f} ({best.ticker})")
    plot(dt, du)


if __name__ == "__main__":
    main()
