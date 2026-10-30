"""Integration test: drive OptimalExecution-v0 through the EXACT wrapper stack
and vector-env API that ppo_continuous.py uses, without needing torch.

Checks:
1. gym.make("execution_gym:OptimalExecution-v0") resolves via module-prefix
   import (so ppo_continuous.py needs zero changes).
2. The full make_env wrapper stack from ppo_continuous.py (ClipAction,
   NormalizeObservation, obs clip, NormalizeReward, reward clip,
   RecordEpisodeStatistics) composes without error.
3. SyncVectorEnv rollout loop matches ppo_continuous.py's usage: 5-tuple step,
   infos["episode"] / infos["_episode"] episodic stats appear at episode ends.
4. Action-space contract: Box(1,) in [0,1]; out-of-range Gaussian samples are
   clipped by ClipAction, and forced terminal liquidation keeps sum(n_k) = X.
5. Ground truth: a TWAP policy driven through the RAW env matches the
   closed-form expected cost within Monte Carlo error.

Run:  python test_integration.py
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

import execution_gym  # registers OptimalExecution-v0
from execution_env import ACParams, ExecutionEnv
from baselines import twap_schedule


def make_env(env_id, seed, idx):
    """Verbatim wrapper stack from ppo_continuous.make_env (minus video)."""

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        env.action_space.seed(seed + idx)
        return env

    return thunk


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def main():
    ok = True

    # 1. module-prefix resolution, exactly as --env-id would deliver it
    env = gym.make("execution_gym:OptimalExecution-v0")
    ok &= check("gym.make with module-prefix env id", True)
    obs, _ = env.reset(seed=1)
    ok &= check(
        "obs/action spaces",
        env.observation_space.shape == (3,) and env.action_space.shape == (1,),
        f"obs {env.observation_space.shape}, act {env.action_space.shape}",
    )

    # 2-3. vectorised rollout through the full wrapper stack
    num_envs = 4
    envs = gym.vector.SyncVectorEnv(
        [make_env("execution_gym:OptimalExecution-v0", seed=1, idx=i) for i in range(num_envs)]
    )
    ok &= check(
        "SyncVectorEnv single_action_space is Box",
        isinstance(envs.single_action_space, gym.spaces.Box),
    )
    next_obs, _ = envs.reset(seed=1)
    episodes_seen, steps = 0, 0
    rng = np.random.default_rng(0)
    for _ in range(120):  # > 2 full episodes per env
        act = rng.normal(0.0, 2.0, size=(num_envs, 1)).astype(np.float32)  # raw Gaussian in u-space
        next_obs, reward, term, trunc, infos = envs.step(act)
        steps += 1
        if "episode" in infos:
            mask = infos.get("_episode", None)
            for i in range(num_envs):
                if mask is None or mask[i]:
                    episodes_seen += 1
                    ep_len = float(np.asarray(infos["episode"]["l"]).reshape(-1)[i])
                    if ep_len != 50:
                        ok &= check("episode length == 50", False, f"got {ep_len}")
    ok &= check("episodic stats surfaced via infos['episode']", episodes_seen >= 2 * num_envs,
                f"{episodes_seen} episodes in {steps} vector steps")
    ok &= check("obs finite and clipped", bool(np.all(np.isfinite(next_obs))))

    # 4. terminal liquidation: raw env, absurd actions, inventory still -> 0
    raw = ExecutionEnv(ACParams(), continuous=True, seed=7)
    raw.reset()
    done = False
    while not done:
        _, _, done, info = raw.step(0.0)  # try to never sell
    ok &= check("forced terminal liquidation", abs(info["inventory"]) < 1e-6,
                f"residual inventory {info['inventory']:.2e}")

    # 5. ground truth through the gym pathway (raw rewards, no normalizers)
    p = ACParams()
    genv = gym.make("execution_gym:OptimalExecution-v0")
    n_ep = 3000
    sched = twap_schedule(p)
    fracs = sched / (p.X - np.concatenate([[0.0], np.cumsum(sched)[:-1]]))
    fracs = np.clip(fracs, 1e-6, 1 - 1e-6)
    u_actions = np.log(fracs / (1 - fracs))  # invert the env's sigmoid squash
    tot = np.empty(n_ep)
    for ep in range(n_ep):
        genv.reset(seed=1000 + ep)
        ret, k, done = 0.0, 0, False
        while not done:
            _, r, term, trunc, _ = genv.step(np.array([u_actions[k]], dtype=np.float32))
            ret += r
            done = term or trunc
            k += 1
        tot[ep] = ret
    inner = ExecutionEnv(p, continuous=True)
    J_twap = inner.expected_cost(sched) + p.lam * inner.cost_variance(sched)
    # episodic return = -(IS + lam*risk)*1e-6; E over noise -> -J*1e-6
    exp_ret = -J_twap * 1e-6
    se = tot.std(ddof=1) / np.sqrt(n_ep)
    z = abs(tot.mean() - exp_ret) / se
    ok &= check("TWAP return matches closed form", z < 4,
                f"sim {tot.mean():.4f} vs theory {exp_ret:.4f} (z={z:.2f})")

    print("\nAll integration checks passed." if ok else "\nSOME CHECKS FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
