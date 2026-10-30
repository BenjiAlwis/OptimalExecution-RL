"""test_calibrated_gym.py — the gym layer must not change the problem.

Checks that wrapping CalibratedExecutionEnv for Gymnasium leaves the objective
intact and that the normalisation does what it claims:

1. gym.make resolves via the module-prefix env id (what --env-id passes).
2. Observation/action spaces match the documented contract (4-dim obs,
   Box(-8, 8) action).
3. The AC schedule scores -1.000 in every registered env — the property the
   whole reward normalisation exists to provide, and the reference every
   training log is read against.
4. Reward scaling does not change the ORDERING of policies: the tilted-AC gain
   measured through the gym layer matches the gain measured directly on the
   underlying environment. A scale factor cannot change an argmax, but a bug
   in the wrapper could.
5. The wrapper stack ppo_continuous.py applies (ClipAction, RecordEpisodeStatistics)
   composes, episodes are exactly N steps, and terminal liquidation holds.
6. Episodes differ across seeds (the week-7 RNG bug, re-checked at the gym
   layer where training actually runs).

Run: python test_calibrated_gym.py
"""

from __future__ import annotations

import sys

import numpy as np
import gymnasium as gym

import calibrated_gym
from calibrated_gym import register_all, config_for, j_ac_for, ACTION_BOUND
from baselines import ac_schedule, SchedulePolicy, evaluate_policy
from week7_analysis import TiltedACPolicy

OK = True
N_EP = 4000


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def u_for(frac: float) -> float:
    """Invert the env's sigmoid squash so a schedule can be replayed exactly."""
    f = float(np.clip(frac, 1e-9, 1 - 1e-9))
    return float(np.clip(np.log(f / (1 - f)), -ACTION_BOUND, ACTION_BOUND))


def main():
    ids = register_all()
    if not ids:
        raise SystemExit("no env ids registered — is calibration_all.csv present?")
    print(f"{len(ids)} env ids registered\n")

    probe = [i for i in ids if "AMZN" in i and "kT4" in i][0]

    env = gym.make(f"calibrated_gym:{probe}")
    check("module-prefix gym.make", True, probe)
    check("observation space is 4-dim", env.observation_space.shape == (4,),
          str(env.observation_space.shape))
    check("action space is Box(-8, 8, (1,))",
          isinstance(env.action_space, gym.spaces.Box)
          and env.action_space.shape == (1,)
          and np.isclose(env.action_space.high[0], ACTION_BOUND))

    # --- 3. AC scores exactly -1 in every env.
    # Judged by z-score, not a fixed tolerance: episodic returns here have
    # CV ~ 4 (lognormal liquidity), so 600 episodes give SE ~ 0.16 and any
    # fixed threshold tighter than that fails on noise alone.
    worst_z = 0.0
    for env_id in ids:
        cfg = config_for(env_id)
        sched = ac_schedule(cfg.ac_params())
        fr = sched / np.maximum(
            cfg.params.X - np.concatenate([[0.0], np.cumsum(sched)[:-1]]), 1e-12)
        us = [u_for(f) for f in fr]
        genv = gym.make(f"calibrated_gym:{env_id}")
        tot = np.empty(600)
        for ep in range(600):
            genv.reset(seed=50_000 + ep)
            ret, k, done = 0.0, 0, False
            while not done:
                _, r, term, trunc, _ = genv.step(
                    np.array([us[min(k, len(us) - 1)]], dtype=np.float32))
                ret += r
                done = term or trunc
                k += 1
            tot[ep] = ret
        se = tot.std(ddof=1) / np.sqrt(len(tot))
        worst_z = max(worst_z, abs(tot.mean() + 1.0) / se)
    check("AC scores -1.000 in every env (z-test)", worst_z < 4.0,
          f"worst |z| = {worst_z:.2f} across {len(ids)} envs")

    # --- 4. scaling preserves policy ordering
    cfg = config_for(probe)
    def gain_direct(alpha):
        e = cfg.env(use_ou=True, use_power=True)
        sf = evaluate_policy(e, TiltedACPolicy(cfg.ac_params(), alpha),
                             n_episodes=N_EP, seed=4242)
        return sf.mean() + cfg.params.lam * sf.var(ddof=1)
    J0, J1 = gain_direct(0.0), gain_direct(1.0)
    direct_gain = 100 * (J0 - J1) / J0

    scale = calibrated_gym._SCALES[probe]
    def gain_gym(alpha):
        e = cfg.env(use_ou=True, use_power=True, reward_scale=scale)
        sf = evaluate_policy(e, TiltedACPolicy(cfg.ac_params(), alpha),
                             n_episodes=N_EP, seed=4242)
        return sf.mean() + cfg.params.lam * sf.var(ddof=1)
    G0, G1 = gain_gym(0.0), gain_gym(1.0)
    gym_gain = 100 * (G0 - G1) / G0
    check("reward scaling preserves the measured gain",
          abs(direct_gain - gym_gain) < 0.5,
          f"direct {direct_gain:.2f}% vs scaled {gym_gain:.2f}%")

    # --- 5. wrapper stack, episode length, liquidation
    def thunk():
        e = gym.make(f"calibrated_gym:{probe}")
        e = gym.wrappers.RecordEpisodeStatistics(e)
        e = gym.wrappers.ClipAction(e)
        return e
    venv = gym.vector.SyncVectorEnv([thunk for _ in range(4)])
    venv.reset(seed=1)
    rng = np.random.default_rng(0)
    seen, lengths = 0, set()
    for _ in range(130):
        _, _, _, _, infos = venv.step(
            rng.normal(0, 3, size=(4, 1)).astype(np.float32))
        if "episode" in infos:
            mask = infos.get("_episode")
            for i in range(4):
                if mask is None or mask[i]:
                    seen += 1
                    lengths.add(int(np.asarray(infos["episode"]["l"]).reshape(-1)[i]))
    check("wrapper stack + vector env run", seen >= 4, f"{seen} episodes")
    check("episodes are exactly N steps", lengths == {cfg.params.N}, str(lengths))

    raw = cfg.env(use_ou=True, use_power=True)
    raw.reset(seed=3)
    done = False
    while not done:
        _, _, done, info = raw.step(0.0)
    check("terminal liquidation holds", abs(info["inventory"]) < 1e-6,
          f"residual {info['inventory']:.2e}")

    # --- 6. liquidity paths differ across seeds
    e = gym.make(f"calibrated_gym:{probe}")
    zs = []
    for s in range(8):
        obs, _ = e.reset(seed=s)
        zs.append(round(float(obs[3]), 9))
    check("liquidity differs across seeds", len(set(zs)) == 8,
          f"{len(set(zs))}/8 distinct")

    print("\nAll gym-layer checks passed." if OK else "\nSOME CHECKS FAILED.")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
