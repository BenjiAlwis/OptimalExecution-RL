"""calibrated_env.py — one environment class configured from real calibration.

Week 5 used three separate variant subclasses, each breaking one Almgren-Chriss
assumption in isolation. Week 7 needs them combined: the LOBSTER cross-section
says AMZN/GOOG/AAPL have concave books (fitted exponent 0.53-0.61) AND
mean-reverting liquidity (theta 34-63), so the right environment for those
names has both at once.

`CalibratedExecutionEnv` generalises both:

    per-share temporary impact = eps + c_t * (n / tau) ** power
    log c_t follows an OU process with autocorrelation exp(-theta*tau) and
    stationary std `log_std`, calibrated so E[c_t] = c exactly.

    power = 1, log_std = 0  ->  base ExecutionEnv (bit-exact; see tests)
    power = 1, log_std > 0  ->  week-5 StochasticLiquidityEnv
    power < 1, log_std = 0  ->  week-5 SqrtImpactEnv

IMPACT COEFFICIENT CALIBRATION
------------------------------
`c` is set so the model reproduces the empirical square-root law at the order
size actually being executed:

    dP/P = Y * sigma_rel * sqrt(X / ADV)          (Y ~ 0.5)
    c = dP_per_share / (X / T) ** power

This matters because the book-walk estimate of the impact COEFFICIENT came out
37-293x below the law (it prices an instant sweep, not an order worked over
hours), while the fitted EXPONENT from the same book walk is the informative
quantity. So week 7 takes the exponent from the book and the level from the
law. Note the linearised coefficient depends on X when power != the law's 0.5,
so `c` must be recomputed whenever X changes — `build_from_calibration` does.

The AC baseline for every ticker is the LINEAR schedule computed with the
law-matched linear eta: that is what a desk running the standard formula would
actually trade, and its shortfall in a concave or stochastic environment is the
quantity of interest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from execution_env import ACParams, ExecutionEnv

SQRT_LAW_Y = 0.5
SECONDS_PER_TRADING_DAY = 23_400.0


def sqrt_law_cost_per_share(X: float, adv: float, sigma: float, S0: float,
                            Y: float = SQRT_LAW_Y) -> float:
    """Empirical impact of executing X shares, in $/share."""
    return float(Y * (sigma / S0) * np.sqrt(X / adv) * S0)


def impact_coef_for_power(X: float, adv: float, sigma: float, S0: float,
                          power: float, T: float = 1.0,
                          Y: float = SQRT_LAW_Y) -> float:
    """c such that c*(X/T)**power equals the square-root-law cost per share."""
    dP = sqrt_law_cost_per_share(X, adv, sigma, S0, Y)
    return float(dP / (X / T) ** power)


def linear_eta_for(X: float, adv: float, sigma: float, S0: float,
                   T: float = 1.0, Y: float = SQRT_LAW_Y) -> float:
    """Law-matched LINEAR eta (power = 1) — the coefficient the AC baseline uses."""
    return impact_coef_for_power(X, adv, sigma, S0, 1.0, T, Y)


class CalibratedExecutionEnv(ExecutionEnv):
    """Power-law impact with optional lognormal-OU liquidity."""

    def __init__(self, params: ACParams, power: float = 1.0,
                 impact_coef: float | None = None,
                 theta: float | None = None, log_std: float = 0.0,
                 state_seed: int | None = None, **kwargs):
        self.power = float(power)
        # default keeps the base model exactly: c = eta, power = 1
        self._impact_coef = (float(impact_coef) if impact_coef is not None
                             else float(params.eta))
        self.theta = theta
        self.log_std = float(log_std)
        self._state_seed = state_seed
        self.observation_dim = 4 if log_std > 0 else 3
        super().__init__(params, **kwargs)

    # ------------------------------------------------------------- state
    def reset(self, seed: int | None = None):
        # The liquidity stream must differ ACROSS EPISODES, or every episode
        # replays one path and an agent can memorise it. Deriving the stream
        # from state_seed alone (the week-5 pattern, never exercised because
        # state_seed defaulted to None) made all episodes identical.
        if seed is not None:
            base = self._state_seed if self._state_seed is not None else 777_001
            self.state_rng = np.random.default_rng(seed + base)
        elif not hasattr(self, "state_rng"):
            self.state_rng = np.random.default_rng(self._state_seed)
        p = self.p
        if self.theta is not None and self.log_std > 0:
            self.rho = float(np.exp(-self.theta * p.tau))
            self.z = float(self.state_rng.standard_normal())
        else:
            self.rho, self.z = 0.0, 0.0
        return super().reset(seed=seed)

    @property
    def coef_now(self) -> float:
        """c_t, with E[c_t] = c exactly (lognormal mean correction)."""
        if self.log_std <= 0:
            return self._impact_coef
        return float(self._impact_coef
                     * np.exp(self.log_std * self.z - 0.5 * self.log_std**2))

    # ------------------------------------------------------------- hooks
    def _temp_impact_per_share(self, n_k: float) -> float:
        p = self.p
        if n_k <= 0:
            return 0.0
        if self.power == 1.0:
            # identical expression to the base env so degeneracy is BIT-exact,
            # not merely close: (n/tau)**1.0 reorders the floating-point ops
            return p.eps * np.sign(n_k) + (self.coef_now / p.tau) * n_k
        return p.eps + self.coef_now * (n_k / p.tau) ** self.power

    def _post_trade_update(self, n_k: float) -> None:
        if self.log_std > 0 and self.theta is not None:
            self.z = (self.rho * self.z
                      + np.sqrt(max(1 - self.rho**2, 0.0))
                      * self.state_rng.standard_normal())

    def _extra_obs(self) -> list:
        return [self.z] if self.log_std > 0 else []

    # --------------------------------------------------- closed forms
    def expected_cost_power(self, trade_list: np.ndarray) -> float:
        """E[IS] for a deterministic schedule.

        Valid with OU liquidity too: E[c_t] = c and the schedule is
        deterministic, so expectation passes through the impact term.
        """
        p = self.p
        n = np.asarray(trade_list, dtype=float)
        pos = n[n > 0]
        temp = float((self._impact_coef * pos * (pos / p.tau) ** self.power).sum())
        return 0.5 * p.gamma * (p.X**2 - (n**2).sum()) + p.eps * n.sum() + temp

    def J(self, trade_list: np.ndarray) -> float:
        return (self.expected_cost_power(trade_list)
                + self.p.lam * self.cost_variance(trade_list))


# ------------------------------------------------------------- builder
@dataclass
class TickerConfig:
    ticker: str
    params: ACParams
    power: float
    impact_coef: float
    theta: float
    log_std: float
    X: float
    pct_adv: float
    kappa_T: float
    linear_eta: float          # law-matched linear eta, used by the AC baseline

    def env(self, use_ou: bool = True, use_power: bool = True, **kwargs):
        return CalibratedExecutionEnv(
            self.params,
            power=self.power if use_power else 1.0,
            impact_coef=self.impact_coef if use_power else self.linear_eta,
            theta=self.theta if use_ou else None,
            log_std=self.log_std if use_ou else 0.0,
            continuous=True, **kwargs)

    def ac_params(self) -> ACParams:
        """Parameters a desk running the AC formula would use: LINEAR eta."""
        p = self.params
        return ACParams(X=p.X, S0=p.S0, T=p.T, N=p.N, sigma=p.sigma,
                        gamma=p.gamma, eta=self.linear_eta, eps=p.eps, lam=p.lam)


def build_from_calibration(csv_path: str = "calibration_all.csv",
                           pct_adv: float = 1.0, N: int = 50,
                           lam: float | None = None,
                           target_kappa_T: float | None = None,
                           use_fitted_power: bool = True,
                           Y: float = SQRT_LAW_Y) -> dict[str, TickerConfig]:
    """One TickerConfig per row of the calibration table.

    lam: fixed risk aversion, or None with target_kappa_T to solve for it.
    use_fitted_power: take the exponent from the book-walk fit (per ticker);
        False forces linear impact everywhere.
    """
    df = pd.read_csv(csv_path)
    need = {"ticker", "mid_px", "sigma", "eps", "gamma", "adv", "power",
            "theta", "ou_log_std"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns {sorted(missing)}")

    out: dict[str, TickerConfig] = {}
    for r in df.itertuples():
        X = r.adv * pct_adv / 100.0
        power = float(r.power) if use_fitted_power else 1.0
        c = impact_coef_for_power(X, r.adv, r.sigma, r.mid_px, power, 1.0, Y)
        eta_lin = linear_eta_for(X, r.adv, r.sigma, r.mid_px, 1.0, Y)

        tau = 1.0 / N
        eta_tilde = eta_lin - 0.5 * r.gamma * tau
        if eta_tilde <= 0:
            print(f"[skip] {r.ticker}: eta_tilde = {eta_tilde:.3g} <= 0 even with "
                  "the law-matched eta; AC has no closed form here")
            continue
        if lam is not None:
            lam_use = lam
        elif target_kappa_T is not None:
            lam_use = target_kappa_T**2 * eta_tilde / r.sigma**2
        else:
            lam_use = 2e-6

        p = ACParams(X=X, S0=r.mid_px, T=1.0, N=N, sigma=r.sigma,
                     gamma=r.gamma, eta=eta_lin, eps=r.eps, lam=lam_use)
        out[r.ticker] = TickerConfig(
            ticker=r.ticker, params=p, power=power, impact_coef=c,
            theta=float(r.theta), log_std=float(r.ou_log_std),
            X=X, pct_adv=pct_adv, kappa_T=float(p.kappa * p.T),
            linear_eta=eta_lin)
    return out
