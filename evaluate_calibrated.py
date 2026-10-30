"""evaluate_calibrated.py — score week-7 RL runs against their own baselines.

Reports the statistic weeks 5-7 established as the honest one:
MEDIAN over late checkpoints, not the final checkpoint and not the mean.

  * final-checkpoint scores are a sample from whatever distribution late
    training produces; in one week-5 environment that distribution spanned
    37 percentage points (report section 6.6);
  * the distribution is left-skewed, so rare blow-ups dominate the mean while
    the median tracks typical behaviour. Blow-up frequency is reported
    separately rather than folded into the headline number.

Each policy is scored by Monte Carlo in its own ticker's environment, with
common random numbers across policies, against three references:

    TWAP            naive
    AC              what a desk running the formula would trade
    tilted-AC(a*)   the best one-parameter adaptive rule (week-7 bound)

The question the table answers: does a learned policy exceed the bound a
hand-designed rule reaches? Week 5 got close but did not clear it (6.65%
against 7.75%); here the bounds are ~12% on the concave-book names at high
urgency, where state-dependent responsiveness has the most room to add.

Usage:
    python evaluate_calibrated.py                  # all OE-* runs with the cal tag
    python evaluate_calibrated.py --tag cal --n-episodes 20000
    python evaluate_calibrated.py --env OE-AMZN-kT8-v0
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from calibrated_gym import register_all, config_for, ACTION_BOUND
from baselines import ac_schedule, twap_schedule, SchedulePolicy, evaluate_policy
from week7_analysis import TiltedACPolicy, ALPHAS
from ppo_continuous import Agent

BLOWUP_VS_AC = -5.0     # a checkpoint this far below AC (%) counts as a blow-up
LATE_FRACTION = 0.5


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


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


def load_agent(ckpt: str, obs_dim: int) -> Agent:
    a = Agent(obs_dim=obs_dim, act_dim=1)
    a.load_state_dict(torch.load(ckpt, map_location="cpu"))
    a.eval()
    return a


def mc_J(cfg, policy_factory, n_ep: int, seed: int = 8888) -> float:
    env = cfg.env(use_ou=True, use_power=True)
    sf = evaluate_policy(env, policy_factory(), n_episodes=n_ep, seed=seed)
    return float(sf.mean() + cfg.params.lam * sf.var(ddof=1))


def open_loop_optimum(cfg) -> tuple[np.ndarray, float]:
    """Best DETERMINISTIC schedule for the true dynamics, in closed form.

    This is the baseline that separates the two things a learned policy does.
    AC assumes LINEAR temporary impact; these environments have power 0.53-0.83,
    so the AC schedule has the wrong SHAPE, and correcting that shape requires
    no liquidity information whatsoever. Anything this schedule achieves is
    misspecification correction; only the excess over it is adaptivity.

    Without this baseline the comparison is misleading: at kappa*T = 8 the
    open-loop optimum scores ~15.4% over AC while the tilted-AC rule reaches
    only ~11.5%, so a policy that sees no liquidity at all "beats the bound".
    """
    p = cfg.params
    env = cfg.env(use_ou=True, use_power=True)
    s0 = ac_schedule(cfg.ac_params())
    J0 = env.J(s0)
    res = minimize(lambda n: env.J(np.asarray(n)) / J0, s0, method="SLSQP",
                   bounds=[(0.0, p.X)] * p.N,
                   constraints=[{"type": "eq",
                                 "fun": lambda n: n.sum() / p.X - 1.0}],
                   options={"maxiter": 400, "ftol": 1e-12})
    return res.x, float(env.J(res.x))


def best_tilted(cfg, n_ep: int) -> tuple[float, float]:
    ac_p = cfg.ac_params()
    J_ac = mc_J(cfg, lambda: SchedulePolicy(ac_schedule(ac_p)), n_ep)
    best, ba = J_ac, 0.0
    for a in ALPHAS:
        if a == 0.0:
            continue
        Ja = mc_J(cfg, lambda a=a: TiltedACPolicy(ac_p, a), n_ep)
        if Ja < best:
            best, ba = Ja, a
    return 100.0 * (J_ac - best) / J_ac, ba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cal")
    ap.add_argument("--env", default=None, help="restrict to one env id")
    ap.add_argument("--n-episodes", type=int, default=12000)
    ap.add_argument("--out", default="calibrated_rl_results.csv")
    args = ap.parse_args()

    register_all()
    pattern = f"runs/*OE-*__{args.tag}__*/agent_*.pt"
    ckpts = glob.glob(pattern)
    if not ckpts:
        raise SystemExit(f"no checkpoints matched {pattern} — run run_calibrated.sh first")

    by_env = defaultdict(lambda: defaultdict(list))
    for ck in ckpts:
        run = os.path.basename(os.path.dirname(ck))
        m = re.search(r"(OE-[A-Z]+-kT[\d.]+-v0)", run)
        s = re.search(r"seed(\d+)", run)
        u = re.search(r"agent_(\d+)\.pt", ck)
        if not (m and s and u):
            continue
        if args.env and m.group(1) != args.env:
            continue
        by_env[m.group(1)][int(s.group(1))].append((int(u.group(1)), ck))

    rows = []
    for env_id in sorted(by_env):
        cfg = config_for(env_id)
        obs_dim = cfg.env().observation_dim
        ac_p = cfg.ac_params()
        J_twap = mc_J(cfg, lambda: SchedulePolicy(twap_schedule(ac_p)), args.n_episodes)
        J_ac = mc_J(cfg, lambda: SchedulePolicy(ac_schedule(ac_p)), args.n_episodes)
        bound, alpha_star = best_tilted(cfg, args.n_episodes)
        opt_sched, _ = open_loop_optimum(cfg)
        J_opt = mc_J(cfg, lambda: SchedulePolicy(opt_sched), args.n_episodes)
        shape_gain = 100.0 * (J_ac - J_opt) / J_ac

        print("\n" + "=" * 74)
        print(f"{env_id}   X = {cfg.X:,.0f}   power {cfg.power:.3f}   "
              f"theta {cfg.theta:.1f}   kappa*T {cfg.kappa_T:.2f}")
        print(f"{'TWAP':<24} J = {J_twap:>12,.0f}  ({100*(J_ac-J_twap)/J_ac:+.2f}% vs AC)")
        print(f"{'AC':<24} J = {J_ac:>12,.0f}")
        print(f"{'tilted-AC (a*=%.2f)' % alpha_star:<24} "
              f"J = {J_ac*(1-bound/100):>12,.0f}  ({bound:+.2f}% vs AC)")
        print(f"{'open-loop optimum':<24} J = {J_opt:>12,.0f}  ({shape_gain:+.2f}% vs AC)"
              "  <- SHAPE only, no liquidity info")
        print(f"{'':24} the bar for adaptivity is beating THIS, not AC")

        per_seed = []
        for seed in sorted(by_env[env_id]):
            pts = sorted(by_env[env_id][seed])
            if not pts:
                continue
            cutoff = pts[int(len(pts) * (1 - LATE_FRACTION))][0]
            gains, adapt = [], []
            for upd, ck in pts:
                if upd < cutoff:
                    continue
                J = mc_J(cfg, lambda ck=ck: GreedyPolicy(load_agent(ck, obs_dim)),
                         args.n_episodes)
                gains.append(100.0 * (J_ac - J) / J_ac)
                adapt.append(100.0 * (J_opt - J) / J_opt)
            if not gains:
                continue
            med = float(np.median(gains))
            med_ad = float(np.median(adapt))
            per_seed.append((med, med_ad))
            blow = 100.0 * np.mean(np.array(gains) < BLOWUP_VS_AC)
            print(f"  seed {seed}: vs AC {med:+6.2f}%   vs open-loop opt "
                  f"{med_ad:+6.2f}%  ({len(gains)} ckpts, blow-ups {blow:.0f}%)")
            rows.append(dict(env_id=env_id, seed=seed, median_gain=med,
                             median_adaptivity=med_ad,
                             min_gain=min(gains), max_gain=max(gains),
                             n_ckpts=len(gains), blowup_pct=blow,
                             bound=bound, alpha_star=alpha_star,
                             shape_gain=shape_gain,
                             J_ac=J_ac, J_twap=J_twap, J_open_loop=J_opt))
        if per_seed:
            m = float(np.median([a for a, _ in per_seed]))
            ma = float(np.median([b for _, b in per_seed]))
            verdict = ("captures adaptivity beyond the right shape" if ma > 1.0
                       else "no clear gain over a blind open-loop schedule")
            print(f"  across seeds: vs AC {m:+.2f}%  =  shape {shape_gain:+.2f}% "
                  f"+ adaptivity {ma:+.2f}%  -> {verdict}")
            if ma < bound:
                print(f"    note: adaptivity {ma:+.2f}% is below the tilted-AC "
                      f"rule's {bound:+.2f}%, which rescales a MISSPECIFIED base "
                      "and is not a clean adaptivity comparator")

    if rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
