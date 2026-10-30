"""Optimal execution environment with Almgren-Chriss (2000) price dynamics.

Sell X shares over N intervals of length tau. Discrete arithmetic random walk
with linear permanent and temporary impact:

    S_k       = S_{k-1} + sigma * sqrt(tau) * xi_k - gamma * n_k        (permanent)
    S~_k      = S_{k-1} - (eps * sign(n_k) + (eta / tau) * n_k)         (temporary, execution price)

where n_k >= 0 is the number of shares sold in interval k.

The per-episode objective is implementation shortfall

    IS = X * S_0 - sum_k n_k * S~_k

and the agent is mean-variance: it minimizes E[IS] + lam * Var[IS].
Closed-form results (used by the sanity harness):

    E[IS]   = 0.5 * gamma * X^2 + eps * X + (eta_tilde / tau) * sum_k n_k^2
    Var[IS] = sigma^2 * tau * sum_k x_k^2        (x_k = holdings AFTER trade k)

with eta_tilde = eta - 0.5 * gamma * tau.

The environment follows the classic gym API (reset/step) without requiring the
gymnasium dependency, so it drops into an existing PPO loop with a thin wrapper.

Reward: per-step mean-variance decomposition of the shortfall,

    r_k = -(capture shortfall of trade k) - lam * sigma^2 * tau * x_k^2

which telescopes so that  sum_k r_k = -(IS_realized_drift_free) - lam * (risk term).
Summed over an episode and averaged over noise, maximizing return is equivalent
to minimizing E[IS] + lam * Var[IS] under AC dynamics (the price-noise term has
zero mean and its variance is exactly the penalized term).
"""

from __future__ import annotations

import numpy as np


class ACParams:
    """Almgren-Chriss model parameters (defaults are the values from the paper's
    worked example, scaled to a one-day horizon)."""

    def __init__(
        self,
        X: float = 1_000_000.0,   # total shares to sell
        S0: float = 50.0,         # initial price ($)
        T: float = 1.0,           # horizon (days)
        N: int = 50,              # number of trading intervals
        sigma: float = 0.95,      # daily volatility ($/share/sqrt(day))
        gamma: float = 2.5e-7,    # permanent impact ($/share per share)
        eta: float = 2.5e-6,      # temporary impact ($/share per share/day)
        eps: float = 0.0625,      # fixed cost of selling ($/share), half-spread
        lam: float = 2e-6,        # risk aversion (1/$)
    ):
        self.X, self.S0, self.T, self.N = X, S0, T, N
        self.sigma, self.gamma, self.eta, self.eps, self.lam = sigma, gamma, eta, eps, lam

    @property
    def tau(self) -> float:
        return self.T / self.N

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau

    @property
    def kappa(self) -> float:
        """Discrete-time urgency parameter kappa (AC eq. for the sinh trajectory)."""
        kappa_tilde_sq = self.lam * self.sigma**2 / self.eta_tilde
        # discrete correction: cosh(kappa * tau) = 1 + 0.5 * kappa_tilde^2 * tau^2
        return np.arccosh(0.5 * kappa_tilde_sq * self.tau**2 + 1.0) / self.tau


class ExecutionEnv:
    observation_dim = 3
    """Gym-style sell-side execution environment.

    Observation (np.float32, shape (3,)):
        [0] remaining inventory fraction        x_k / X            in [0, 1]
        [1] remaining time fraction             (N - k) / N        in (0, 1]
        [2] normalized price change since t=0   (S_k - S0) / (sigma * sqrt(T))

    Action:
        discrete mode (default): integer in {0, ..., n_actions-1}, mapped to
        selling that fraction of *remaining* inventory:
            frac = action / (n_actions - 1) * max_frac
        continuous mode: float in [0, 1], fraction of remaining inventory.
        On the final step the entire remaining inventory is liquidated
        regardless of the action (hard terminal constraint).

    Reward (per step, in dollars, scaled by `reward_scale`):
        r_k = -[ n_k * (eps * sign(n_k) + (eta / tau) * n_k)      temporary cost
                 + gamma * n_k * (X - x_{k-1} + 0.5 * n_k) ]      permanent cost (telescoped)
              + n_k * (S_k^drift-free noise term ... folded into capture)
              - lam * sigma^2 * tau * x_k^2                        risk penalty
        implemented directly as (capture - arrival value decrement) so that the
        realized episode return equals -(IS) - lam * sigma^2 * tau * sum x_k^2.
    """

    def __init__(
        self,
        params: ACParams | None = None,
        n_actions: int = 11,
        max_frac: float = 1.0,
        continuous: bool = False,
        reward_scale: float = 1e-4,
        seed: int | None = None,
    ):
        self.p = params or ACParams()
        self.n_actions = n_actions
        self.max_frac = max_frac
        self.continuous = continuous
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.k = 0                      # completed intervals
        self.x = self.p.X               # remaining inventory
        self.S = self.p.S0              # current (permanent-impact-adjusted) price
        self.shortfall = 0.0            # running realized implementation shortfall
        self.risk_term = 0.0            # running sigma^2 tau sum x_k^2
        self.trades: list[float] = []   # n_k history (for diagnostics)
        return self._obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        if self.k >= self.p.N:
            raise RuntimeError("Episode is done; call reset().")
        p = self.p

        # --- map action to shares sold this interval
        if self.k == p.N - 1:
            n_k = self.x  # forced full liquidation on the last step
        else:
            if self.continuous:
                frac = float(np.clip(action, 0.0, 1.0)) * self.max_frac
            else:
                frac = (int(action) / (self.n_actions - 1)) * self.max_frac
            n_k = frac * self.x
        n_k = float(np.clip(n_k, 0.0, self.x))

        # --- execution at temporary-impact-adjusted price (hook for variants)
        exec_price = self.S - self._temp_impact_per_share(n_k)
        capture = n_k * exec_price

        # --- price evolves: noise + permanent impact of this trade
        noise = p.sigma * np.sqrt(p.tau) * self.rng.standard_normal()
        self.S = self.S + noise - p.gamma * n_k
        self._post_trade_update(n_k)

        # --- bookkeeping
        self.x -= n_k
        self.k += 1
        self.trades.append(n_k)

        # incremental shortfall relative to arrival price S0
        step_shortfall = n_k * p.S0 - capture
        self.shortfall += step_shortfall

        # risk penalty on inventory held AFTER this trade
        step_risk = p.sigma**2 * p.tau * self.x**2
        self.risk_term += step_risk

        reward = -(step_shortfall + p.lam * step_risk) * self.reward_scale
        done = self.k >= p.N
        info = {
            "n_k": n_k,
            "shortfall": self.shortfall,
            "risk_term": self.risk_term,
            "inventory": self.x,
        }
        return self._obs(), reward, done, info

    # ------------------------------------------------------------ internals
    def _obs(self) -> np.ndarray:
        p = self.p
        base = [
            self.x / p.X,
            (p.N - self.k) / p.N,
            (self.S - p.S0) / (p.sigma * np.sqrt(p.T)),
        ]
        return np.array(base + self._extra_obs(), dtype=np.float32)

    # ------------------------------------------------- variant hooks
    def _temp_impact_per_share(self, n_k: float) -> float:
        """Per-share temporary impact of selling n_k this interval."""
        p = self.p
        return p.eps * np.sign(n_k) + (p.eta / p.tau) * n_k

    def _post_trade_update(self, n_k: float) -> None:
        """Evolve any variant-specific state after the trade."""

    def _extra_obs(self) -> list:
        """Extra observation features appended by variants."""
        return []

    # ------------------------------------------------- analytical quantities
    def expected_cost(self, trade_list: np.ndarray) -> float:
        """Closed-form E[IS] for a deterministic trade schedule (AC eq. 20-ish)."""
        p = self.p
        n = np.asarray(trade_list, dtype=float)
        return 0.5 * p.gamma * p.X**2 + p.eps * n.sum() + (p.eta_tilde / p.tau) * (n**2).sum()

    def cost_variance(self, trade_list: np.ndarray) -> float:
        """Closed-form Var[IS] for a deterministic trade schedule."""
        p = self.p
        n = np.asarray(trade_list, dtype=float)
        x = p.X - np.cumsum(n)  # holdings after each trade
        return p.sigma**2 * p.tau * (x**2).sum()
