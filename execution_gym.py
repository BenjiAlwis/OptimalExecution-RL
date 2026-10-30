"""Gymnasium adapter for the execution environments — base + week-5 variants.

Registered ids (module-prefix import syntax means ppo_continuous.py needs no
changes):

    execution_gym:OptimalExecution-v0            base Almgren-Chriss dynamics
    execution_gym:OptimalExecutionOU-v0          stochastic OU liquidity (obs 4)
    execution_gym:OptimalExecutionSqrt-v0        square-root impact     (obs 3)
    execution_gym:OptimalExecutionPressure-v0    volume-pressure spread (obs 4)

Action space: Box(-8, 8, (1,)). The env applies a sigmoid squash to map the
unbounded Gaussian output to a (0,1) fraction of remaining inventory — the
week-4 fix for the boundary-clipping pathology (the optimal early-day action
sits at ~0.024, i.e. against a [0,1] clip boundary but at an ordinary interior
point u ~ -3.7 pre-squash). ClipAction to [-8, 8] is harmless.

Recommended flags: --gamma 1.0 --no-clip-vloss, and the make_env normalizer
gate for OptimalExecution* env ids (see README).

Self-check:  python execution_gym.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from execution_env import ACParams, ExecutionEnv
from execution_env_variants import (
    StochasticLiquidityEnv, SqrtImpactEnv, VolumePressureEnv,
)

_VARIANTS = {
    "base": ExecutionEnv,
    "ou": StochasticLiquidityEnv,
    "sqrt": SqrtImpactEnv,
    "pressure": VolumePressureEnv,
}

ACTION_BOUND = 8.0


class ExecutionGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, variant: str = "base", params: ACParams | None = None,
                 reward_scale: float = 1e-6, render_mode=None, **variant_kwargs):
        super().__init__()
        self.p = params or ACParams()
        cls = _VARIANTS[variant]
        self._env = cls(self.p, continuous=True, reward_scale=reward_scale,
                        **variant_kwargs)
        self.render_mode = render_mode
        d = self._env.observation_dim
        low = np.full(d, -np.inf, dtype=np.float32)
        high = np.full(d, np.inf, dtype=np.float32)
        low[0], high[0] = 0.0, 1.0   # inventory fraction
        low[1], high[1] = 0.0, 1.0   # time fraction
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-ACTION_BOUND, high=ACTION_BOUND, shape=(1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
        obs = self._env.reset(seed=seed if seed is not None else None)
        return obs, {}

    def step(self, action):
        u = float(np.asarray(action).reshape(-1)[0])
        frac = 1.0 / (1.0 + np.exp(-u))  # sigmoid squash to (0,1)
        obs, reward, done, info = self._env.step(frac)
        return obs, reward, done, False, info

    def expected_cost(self, schedule):
        return self._env.expected_cost(schedule)

    def cost_variance(self, schedule):
        return self._env.cost_variance(schedule)


def _register():
    specs = {
        "OptimalExecution-v0": "base",
        "OptimalExecutionOU-v0": "ou",
        "OptimalExecutionSqrt-v0": "sqrt",
        "OptimalExecutionPressure-v0": "pressure",
    }
    for env_id, variant in specs.items():
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point="execution_gym:ExecutionGymEnv",
                kwargs={"variant": variant},
            )


_register()


if __name__ == "__main__":
    for env_id in ("OptimalExecution-v0", "OptimalExecutionOU-v0",
                   "OptimalExecutionSqrt-v0", "OptimalExecutionPressure-v0"):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=0)
        tot, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            tot += r
            done = term or trunc
        print(f"{env_id:<32} obs {env.observation_space.shape}  "
              f"random-episode return {tot:8.3f}")
