"""calibrated_gym.py — Gymnasium ids for the week-7 calibrated environments.

Registers one env id per (ticker, urgency) pair, e.g.

    calibrated_gym:OE-AMZN-kT4-v0

so `ppo_continuous.py` trains on it unchanged:

    python ppo_continuous.py --env-id "calibrated_gym:OE-AMZN-kT4-v0" \\
        --mode clip --seed 1 --gamma 1.0 --no-clip-vloss \\
        --total-timesteps 1000000

REWARD NORMALISATION — why it is per ticker
-------------------------------------------
The calibrated objectives differ by orders of magnitude across the
cross-section (INTC executes 104,716 shares of a $27 stock; GOOG 7,061 shares
of a $570 one with ~200x the impact coefficient). A single reward_scale cannot
condition the critic for all of them.

Each env therefore uses reward_scale = 1 / J_AC for its own ticker, which makes
the AC baseline score EXACTLY -1.0. Two consequences worth having:

  * a training log is readable without a lookup table — episodic_return > -1
    means the policy is beating Almgren-Chriss, and -0.88 means beating it by
    12%;
  * the critic sees O(1) targets in every environment.

The scale is a positive constant multiplying the objective, so it cannot change
the optimal policy.

ACTION SPACE
------------
Box(-8, 8, (1,)) with a sigmoid squash inside the env, per the week-4 finding
that the optimal early-day action sits against a [0,1] clip boundary but at an
ordinary interior point (u ~ -3.7) after squashing.

OBSERVATION
-----------
4-dim: [inventory fraction, time-remaining fraction, normalised price change,
liquidity state z]. The liquidity state is what AC cannot see and what the
week-5 opportunism plot showed the policy learning to use.

Requires `calibration_all.csv` in the working directory (from calibrate_all.py).
Self-check:  python calibrated_gym.py
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

from calibrated_env import build_from_calibration, CalibratedExecutionEnv
from baselines import ac_schedule

CALIBRATION_CSV = os.environ.get("OE_CALIBRATION", "calibration_all.csv")
PCT_ADV = float(os.environ.get("OE_PCT_ADV", "1.0"))
N_INTERVALS = int(os.environ.get("OE_N", "50"))
KAPPA_TARGETS = (1.0, 2.0, 4.0, 8.0)
ACTION_BOUND = 8.0

_CONFIGS: dict[str, object] = {}   # env_id -> TickerConfig
_SCALES: dict[str, float] = {}     # env_id -> reward_scale
_J_AC: dict[str, float] = {}       # env_id -> J of the AC baseline


def _j_ac(cfg) -> float:
    """J of the AC schedule in this ticker's environment — CLOSED FORM.

    The episodic reward is a per-step decomposition: shortfall increments that
    telescope to IS, plus a charge of lam*sigma^2*tau*x_k^2 each step. For a
    DETERMINISTIC schedule the expected return is therefore exactly

        -( E[IS] + lam * sigma^2 * tau * sum_k x_k^2 )

    and both terms are closed form (E[c_t] = c, so OU liquidity passes through
    the expectation). Estimating this by Monte Carlo instead was a mistake: the
    shortfall distribution has CV ~ 4 under lognormal liquidity, so 4,000
    episodes left several percent of error, and because every env used the same
    seed stream those errors were CORRELATED — all twenty normalisations were
    off in the same direction, which looked like a systematic bug rather than
    noise. The closed form is exact and makes registration instant.
    """
    env = cfg.env(use_ou=True, use_power=True)
    return float(env.J(ac_schedule(cfg.ac_params())))


class CalibratedGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_id: str, render_mode=None):
        super().__init__()
        cfg = _CONFIGS[env_id]
        self.cfg = cfg
        self._env = cfg.env(use_ou=True, use_power=True,
                            reward_scale=_SCALES[env_id])
        self.render_mode = render_mode
        d = self._env.observation_dim
        low = np.full(d, -np.inf, dtype=np.float32)
        high = np.full(d, np.inf, dtype=np.float32)
        low[0], high[0] = 0.0, 1.0
        low[1], high[1] = 0.0, 1.0
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-ACTION_BOUND, high=ACTION_BOUND,
                                           shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
        return self._env.reset(seed=seed), {}

    def step(self, action):
        u = float(np.asarray(action).reshape(-1)[0])
        frac = 1.0 / (1.0 + np.exp(-u))
        obs, reward, done, info = self._env.step(frac)
        return obs, reward, done, False, info


def register_all(csv_path: str = CALIBRATION_CSV, pct_adv: float = PCT_ADV,
                 n: int = N_INTERVALS, targets=KAPPA_TARGETS,
                 verbose: bool = False) -> list[str]:
    """Register one env id per (ticker, kappa*T). Returns the ids."""
    if not os.path.exists(csv_path):
        if verbose:
            print(f"[calibrated_gym] {csv_path} not found; nothing registered")
        return []

    ids = []
    for kT in targets:
        cfgs = build_from_calibration(csv_path, pct_adv=pct_adv, N=n,
                                      target_kappa_T=kT)
        for tk, cfg in cfgs.items():
            env_id = f"OE-{tk}-kT{kT:g}-v0"
            _CONFIGS[env_id] = cfg
            if env_id not in _J_AC:
                j = _j_ac(cfg)
                _J_AC[env_id] = j
                _SCALES[env_id] = 1.0 / j if j > 0 else 1e-6
            if env_id not in gym.registry:
                gym.register(id=env_id,
                             entry_point="calibrated_gym:CalibratedGymEnv",
                             kwargs={"env_id": env_id})
            ids.append(env_id)
    return ids


def config_for(env_id: str):
    """TickerConfig behind an env id (strips any module prefix)."""
    key = env_id.split(":")[-1]
    if key not in _CONFIGS:
        register_all()
    return _CONFIGS[key]


def j_ac_for(env_id: str) -> float:
    key = env_id.split(":")[-1]
    if key not in _J_AC:
        register_all()
    return _J_AC[key]


_REGISTERED = register_all()


if __name__ == "__main__":
    ids = register_all(verbose=True)
    if not ids:
        raise SystemExit(f"No ids registered — is {CALIBRATION_CSV} present?")
    print(f"registered {len(ids)} env ids at {PCT_ADV}% of volume, N={N_INTERVALS}\n")
    print(f"{'env id':<22} {'X':>9} {'power':>6} {'theta':>6} {'kappa*T':>8} "
          f"{'J_AC':>12} {'reward_scale':>13}")
    for env_id in ids:
        c = _CONFIGS[env_id]
        print(f"{env_id:<22} {c.X:9,.0f} {c.power:6.3f} {c.theta:6.2f} "
              f"{c.kappa_T:8.2f} {_J_AC[env_id]:12,.0f} {_SCALES[env_id]:13.3g}")

    probe = ids[0]
    env = gym.make(f"calibrated_gym:{probe}")
    obs, _ = env.reset(seed=0)
    tot, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        tot += r
        done = term or trunc
    print(f"\n{probe}: obs {env.observation_space.shape}, "
          f"random-policy return {tot:.3f}")
    print("AC baseline scores exactly -1.000 in every env by construction;")
    print("episodic_return > -1 means the policy is beating Almgren-Chriss.")
