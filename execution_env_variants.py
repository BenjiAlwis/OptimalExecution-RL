"""Week-5 environment variants. Each breaks exactly one Almgren-Chriss
assumption, per the pillar analysis:

  P1 quadratic time-separable costs   -> SqrtImpactEnv
  P2 certainty equivalence            -> StochasticLiquidityEnv
  P3 (inventory, time) Markov state   -> VolumePressureEnv

Design rules shared by all three:
- Subclass ExecutionEnv, overriding only the hook methods
  (_temp_impact_per_share, _post_trade_update, _extra_obs), so the reward
  identity (return = -(IS + lam*risk)) and forced terminal liquidation are
  inherited untouched.
- Every variant has a knob setting at which it degenerates to the base
  environment EXACTLY (same-seed, same-action trajectory equality) —
  variant_sanity.py enforces this.
- Variant-specific randomness uses a SEPARATE generator (state_rng), so the
  price-noise stream is unchanged vs. the base env under the same seed, and
  policy comparisons can use common random numbers.

Testable predictions (from the pillar analysis):
- StochasticLiquidityEnv: with E[eta_t] calibrated to the base eta, the AC
  schedule is EXACTLY the optimal open-loop schedule (expected temporary cost
  is linear in eta_t, so for any deterministic schedule it equals the base
  value). Any RL improvement over AC here is therefore a pure measurement of
  the value of adaptivity. RL should beat AC.
- SqrtImpactEnv: no stochastic state beyond unhedgeable price noise, so the
  optimum remains deterministic; RL should MATCH the numerically optimal
  open-loop schedule (analyze_variants.py computes it) and both should beat
  the mis-specified AC schedule.
- VolumePressureEnv: the pressure state is a deterministic function of the
  agent's own past actions, so again the optimum is deterministic; RL should
  MATCH the numeric open-loop optimum. Empirical correction to the original
  prediction (analyze_variants.py): spread-pressure PENALIZES clustering, so
  it rewards smoothness — and at these parameters the numeric optimum
  coincides with AC to 4 significant figures. The variant is therefore a
  robustness result (AC survives spread memory at realistic scale) and a
  third match-target for RL, not a beat-target. Pulsing would arise in
  resilience/transient-impact models (Obizhaeva-Wang), where price
  displacement decays and trading into recovered prices is cheap — a
  different mechanism than a spread that widens with your own volume.
"""

from __future__ import annotations

import numpy as np

from execution_env import ACParams, ExecutionEnv


class StochasticLiquidityEnv(ExecutionEnv):
    """Temporary-impact coefficient eta_t follows a lognormal OU process.

    log eta_t mean-reverts (exact discretization) with per-interval
    autocorrelation rho = exp(-theta*tau) and stationary std `log_std`. The
    stationary mean is set to log(eta) - log_std^2/2 so that E[eta_t] equals
    the base eta exactly (lognormal mean correction). The agent observes the
    standardized log-liquidity state z_t; AC does not.

    log_std = 0 degenerates to the base environment exactly.
    """

    def __init__(self, *args, theta: float = 10.0, log_std: float = 0.4,
                 state_seed: int | None = None, **kwargs):
        self.theta = theta
        self.log_std = log_std
        self._state_seed = state_seed
        super().__init__(*args, **kwargs)

    observation_dim = 4

    # note: base __init__ calls reset(), so state init lives in reset()
    def reset(self, seed: int | None = None):
        if not hasattr(self, "state_rng") or seed is not None:
            s = self._state_seed if self._state_seed is not None else (
                None if seed is None else seed + 777_001)
            self.state_rng = np.random.default_rng(s)
        p = self.p
        self.rho = np.exp(-self.theta * p.tau)
        self.mu_log = np.log(p.eta) - 0.5 * self.log_std**2
        # start at a stationary draw
        self.z = (self.state_rng.standard_normal() if self.log_std > 0 else 0.0)
        return super().reset(seed=seed)

    @property
    def eta_now(self) -> float:
        # eta * exp(log_std*z - log_std^2/2): E[eta_now] = eta exactly, and
        # the log_std=0 degenerate case returns eta bit-exactly (exp(0)=1)
        return float(self.p.eta * np.exp(self.log_std * self.z - 0.5 * self.log_std**2))

    def _temp_impact_per_share(self, n_k: float) -> float:
        p = self.p
        return p.eps * np.sign(n_k) + (self.eta_now / p.tau) * n_k

    def _post_trade_update(self, n_k: float) -> None:
        if self.log_std > 0:
            self.z = self.rho * self.z + np.sqrt(1 - self.rho**2) * \
                self.state_rng.standard_normal()
        # log_std == 0: z stays 0, eta_now == p.eta exactly (degenerate case)

    def _extra_obs(self) -> list:
        return [self.z]


class SqrtImpactEnv(ExecutionEnv):
    """Power-law temporary impact: per-share impact c * (n/tau)^power.

    power = 1 with the calibrated c equals the base env exactly. power = 0.5
    is the variant. c is calibrated so that per-share impact matches the base
    linear model at the TWAP trading rate v* = X/T:

        c = eta * v*^(1 - power)

    so TWAP has identical temporary cost in both models and the comparison
    isolates the *curvature* change: sub-quadratic total cost penalizes large
    trades less and small trades relatively more, so the optimum concentrates
    trading more than sinh.
    """

    def __init__(self, *args, power: float = 0.5, **kwargs):
        self.power = power
        super().__init__(*args, **kwargs)
        # calibrated so per-share impact matches the linear model at TWAP rate
        self.c = self.p.eta * (self.p.X / self.p.T) ** (1.0 - power)

    observation_dim = 3

    def _temp_impact_per_share(self, n_k: float) -> float:
        p = self.p
        if n_k <= 0:
            return 0.0
        return p.eps + self.c * (n_k / p.tau) ** self.power

    def expected_cost_sqrt(self, trade_list: np.ndarray) -> float:
        """E[IS] for a deterministic schedule under power-law impact.

        Derivation mirrors the base case: noise vanishes in expectation, the
        permanent cross-term telescopes to (gamma/2)(X^2 - sum n^2), and the
        temporary term is sum n_k * c * (n_k/tau)^power.
        """
        p = self.p
        n = np.asarray(trade_list, dtype=float)
        pos = n[n > 0]
        temp = (self.c * pos * (pos / p.tau) ** self.power).sum()
        # permanent cross-term telescopes: gamma/2 * (X^2 - sum n^2)
        return 0.5 * p.gamma * (p.X**2 - (n**2).sum()) + p.eps * n.sum() + temp


class VolumePressureEnv(ExecutionEnv):
    """Spread widens under recent volume pressure (non-Markov in (x, t)).

    Pressure state: P_{k+1} = rho_p * P_k + kappa_p * (n_k / X), applied to
    the NEXT trades' spread: eps_k = eps0 * (1 + beta * P_k). Trading now
    raises the near-future cost of trading, with decay rho_p per interval.
    The agent observes P_k; the (x, t) state alone is no longer Markov.

    beta = 0 (or kappa_p = 0) degenerates to the base environment exactly.
    """

    def __init__(self, *args, beta: float = 1.0, rho_p: float = 0.75,
                 kappa_p: float = 10.0, **kwargs):
        self.beta, self.rho_p, self.kappa_p = beta, rho_p, kappa_p
        super().__init__(*args, **kwargs)

    observation_dim = 4

    def reset(self, seed: int | None = None):
        self.P = 0.0
        return super().reset(seed=seed)

    def _temp_impact_per_share(self, n_k: float) -> float:
        p = self.p
        eps_now = p.eps * (1.0 + self.beta * self.P)
        return eps_now * np.sign(n_k) + (p.eta / p.tau) * n_k

    def _post_trade_update(self, n_k: float) -> None:
        self.P = self.rho_p * self.P + self.kappa_p * (n_k / self.p.X)

    def _extra_obs(self) -> list:
        return [self.P]

    def expected_cost_pressure(self, trade_list: np.ndarray) -> float:
        """E[IS] for a deterministic schedule (pressure path is deterministic)."""
        p = self.p
        n = np.asarray(trade_list, dtype=float)
        P, temp = 0.0, 0.0
        for nk in n:
            eps_now = p.eps * (1.0 + self.beta * P)
            if nk > 0:
                temp += nk * (eps_now + (p.eta / p.tau) * nk)
            P = self.rho_p * P + self.kappa_p * (nk / p.X)
        return 0.5 * p.gamma * (p.X**2 - (n**2).sum()) + temp
