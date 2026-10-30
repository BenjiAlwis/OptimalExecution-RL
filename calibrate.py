"""calibrate.py — estimate Almgren-Chriss parameters from LOBSTER data.

Each estimator targets one parameter of the model family used in weeks 1-5,
and each is validated against synthetic data with known ground truth in
test_calibration.py before being trusted on real prices.

    sigma   midprice volatility, $/share/sqrt(day)
              from log-free arithmetic returns on a clock-time grid,
              scaled to the project's one-day horizon.

    eps     half-spread, $/share
              time-weighted mean of (ask - bid)/2. This is the fixed
              per-share cost in the AC execution price.

    eta     temporary impact, $/share per (share/day)
              from WALKING THE BOOK: for a grid of order sizes n, compute
              the true cost per share of consuming the bid ladder, then fit
                  cost_per_share(n) = a + c * (n / tau)^power
              Fitting `power` as a free parameter is the empirical test of
              week 5's P1 question: power ~ 1 supports the linear AC model,
              power ~ 0.5 supports the square-root variant. With power
              pinned to 1, c = eta.

              TWO PITFALLS, both found by the synthetic recovery tests and
              both silent on real data:
              (a) Probe sizes below top-of-book depth do not move the price
                  at all — the walk stays inside level 1 and the cost is
                  exactly the half-spread. Including those points adds a
                  FLAT segment that biases the exponent upward (measured
                  1.56 instead of 1.0 on a known-linear book). Probe sizes
                  are therefore chosen as multiples of the median level-1
                  depth, starting at 2x.
              (b) The intercept `a` of the walk-cost curve is NOT the
                  half-spread. For a uniform ladder the asymptotic intercept
                  is eps - tick/2, because the average fill price sits half
                  a tick inside the marginal level. Pinning a = eps forces
                  the curvature into `power` and biases eta downward (-32%
                  on a known book). `a` is fitted freely; the separately
                  measured half-spread is reported alongside it as
                  `eps_observed` and is what feeds ACParams, since the AC
                  model's eps is the cost of an infinitesimal trade.

    gamma   permanent impact, $/share per share
              Kyle-style regression of interval midprice changes on net
              signed order flow:  dMid_k = gamma * Q_k + noise.
              Sign convention handled in lobster_io.signed_flow().

    theta, log_std   OU parameters of the liquidity state
              Estimate eta_t on a coarse grid (e.g. 5-minute buckets), take
              logs, fit an AR(1):  log eta_{k+1} = a + rho * log eta_k + e.
              Then theta = -ln(rho) / dt_days and the stationary std is
              std(e) / sqrt(1 - rho^2). These feed StochasticLiquidityEnv
              directly.

CLI:
    python calibrate.py --messages AMZN_..._message_10.csv \\
                        --orderbook AMZN_..._orderbook_10.csv --levels 10
    python calibrate.py --self-test          # synthetic recovery check
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning

from lobster_io import LOBData, load_lobster, resample_grid

SECONDS_PER_TRADING_DAY = 23_400.0  # 09:30-16:00


@dataclass
class Calibration:
    """Estimated parameters, in the project's units (one-day horizon)."""

    sigma: float                 # $/share/sqrt(day)
    eps: float                   # $/share
    eta: float                   # $/share per (share/day), linear fit
    gamma: float                 # $/share per share
    impact_power: float          # free-exponent fit
    impact_coef: float           # c in the free-exponent fit
    impact_power_stderr: float
    impact_intercept: float = np.nan   # fitted `a`; differs from eps by ~tick/2
    gamma_se: float = np.nan           # OLS standard error; gamma is often unidentified
    gamma_r2: float = np.nan
    probe_min: float = np.nan          # smallest/largest order size actually fitted;
    probe_max: float = np.nan          # extrapolating far beyond probe_max is invalid
    adv: float = np.nan                # session traded volume, shares
    l1_depth: float = np.nan
    theta: float = np.nan        # OU mean reversion, 1/day
    ou_log_std: float = np.nan   # stationary std of log eta
    n_snapshots: int = 0
    ticker: str = ""
    date: str = ""

    def to_acparams_snippet(self, X: float = 1e6, S0: float | None = None,
                            N: int = 50, lam: float = 2e-6) -> str:
        return (
            "ACParams(\n"
            f"    X={X:.0f}, S0={S0 if S0 else 'None'}, T=1.0, N={N},\n"
            f"    sigma={self.sigma:.6g}, gamma={self.gamma:.6g},\n"
            f"    eta={self.eta:.6g}, eps={self.eps:.6g}, lam={lam:.6g},\n"
            ")"
        )


# --------------------------------------------------------------- sigma
def estimate_sigma(grid: pd.DataFrame, interval_s: float) -> float:
    """Arithmetic midprice volatility scaled to $/share/sqrt(day)."""
    d = np.diff(grid["mid"].to_numpy())
    d = d[np.isfinite(d)]
    per_interval = d.std(ddof=1)
    return float(per_interval * np.sqrt(SECONDS_PER_TRADING_DAY / interval_s))


# ----------------------------------------------------------------- eps
def estimate_eps(lob: LOBData) -> float:
    """Time-weighted mean half-spread ($/share)."""
    t = lob.time
    hs = 0.5 * lob.spread
    dt = np.diff(t)
    ok = np.isfinite(dt) & (dt >= 0) & np.isfinite(hs[:-1])
    if ok.sum() == 0 or dt[ok].sum() == 0:
        return float(np.nanmean(hs))
    return float(np.sum(hs[:-1][ok] * dt[ok]) / np.sum(dt[ok]))


# ------------------------------------------------------- walking the book
def walk_book_cost(prices: np.ndarray, sizes: np.ndarray, mid: float,
                   n_shares: float, side: str = "sell") -> float:
    """Cost per share (in $) of executing n_shares against a ladder.

    For a sell, consume bid levels best (highest) first; cost per share is
    mid - average execution price. Returns nan if the ladder is too thin to
    fill the order — the caller must drop those points rather than
    extrapolating, since a partially-filled walk understates impact.
    """
    if n_shares <= 0 or len(prices) == 0:
        return np.nan
    order = np.argsort(prices)[::-1] if side == "sell" else np.argsort(prices)
    px, sz = prices[order], sizes[order]
    cum = np.cumsum(sz)
    if cum[-1] < n_shares:
        return np.nan
    k = int(np.searchsorted(cum, n_shares))
    filled = np.minimum(sz[: k + 1], n_shares - np.concatenate([[0.0], cum[:k]]))
    avg_px = float(np.sum(px[: k + 1] * filled) / n_shares)
    return (mid - avg_px) if side == "sell" else (avg_px - mid)


def impact_curve(lob: LOBData, sizes: np.ndarray, max_snapshots: int = 4000,
                 side: str = "sell", seed: int = 0) -> pd.DataFrame:
    """Mean cost-per-share at each order size, averaged over snapshots."""
    rng = np.random.default_rng(seed)
    n = len(lob.book)
    idx = np.sort(rng.choice(n, size=min(max_snapshots, n), replace=False))
    mid = lob.mid
    rows = []
    for i in idx:
        px, sz = (lob.bid_ladder(i) if side == "sell" else lob.ask_ladder(i))
        if len(px) == 0:
            continue
        for s in sizes:
            c = walk_book_cost(px, sz, mid[i], s, side)
            if np.isfinite(c):
                rows.append((s, c))
    df = pd.DataFrame(rows, columns=["n_shares", "cost_per_share"])
    out = df.groupby("n_shares", as_index=False).agg(
        cost_per_share=("cost_per_share", "mean"),
        cost_sd=("cost_per_share", "std"),
        n_obs=("cost_per_share", "size"),
    )
    # SE of the mean at each size: without it, curve_fit treats the averaged
    # points as exact and reports an exponent CI an order of magnitude too
    # tight (1.053 +/- 0.0115 "rejecting" a book that is exactly linear).
    out["cost_se"] = out["cost_sd"] / np.sqrt(out["n_obs"].clip(lower=1))
    # A probe larger than the displayed book returns nan, so it is measured
    # ONLY on snapshots that happened to be deep -- i.e. cheap. Left in, that
    # selection flattens the curve and collapses the fitted exponent toward 0
    # (observed: 0.0001 on a known-linear book). fill_rate exposes it.
    out["fill_rate"] = out["n_obs"] / max(len(idx), 1)
    return out


def fit_impact(curve: pd.DataFrame, tau_days: float,
               eps_fixed: float | None = None, use_weights: bool = True) -> dict:
    """Fit cost_per_share = eps + c * (n/tau)^power, and the power=1 case.

    Returns both: the free-exponent fit (whose `power` is the empirical
    linear-vs-sqrt verdict) and the linear fit that yields eta for ACParams.
    """
    n = curve["n_shares"].to_numpy(float)
    y = curve["cost_per_share"].to_numpy(float)
    rate = n / tau_days
    sig = None
    if use_weights and "cost_se" in curve.columns:
        se = curve["cost_se"].to_numpy(float)
        if np.all(np.isfinite(se)) and np.all(se > 0):
            sig = se

    def model_free(r, eps, c, power):
        return eps + c * r**power

    def model_lin(r, eps, c):
        return eps + c * r

    p0_c = max((y.max() - y.min()) / max(rate.max(), 1e-12), 1e-12)
    # a perfect fit (synthetic data) leaves the covariance undefined; that is
    # not an error and pcov is only used for the exponent's standard error
    warnings.simplefilter("ignore", OptimizeWarning)

    if eps_fixed is None:
        popt_f, pcov_f = curve_fit(model_free, rate, y, sigma=sig,
                                   absolute_sigma=sig is not None,
                                   p0=[y.min(), p0_c, 1.0], maxfev=20000)
        popt_l, _ = curve_fit(model_lin, rate, y, p0=[y.min(), p0_c], maxfev=20000)
        eps_f, c_f, power = popt_f
        eps_l, eta = popt_l
    else:
        popt_f, pcov_f = curve_fit(lambda r, c, p: eps_fixed + c * r**p, rate, y,
                                   sigma=sig, absolute_sigma=sig is not None,
                                   p0=[p0_c, 1.0], maxfev=20000)
        c_f, power = popt_f
        eps_f = eps_fixed
        popt_l, _ = curve_fit(lambda r, c: eps_fixed + c * r, rate, y,
                              p0=[p0_c], maxfev=20000)
        eta = popt_l[0]
        eps_l = eps_fixed
    perr = np.sqrt(np.diag(pcov_f))
    return {
        "eps_from_fit": float(eps_f),
        "impact_coef": float(c_f),
        "impact_power": float(power),
        "impact_power_stderr": float(perr[-1]),
        "eta_linear": float(eta),
        "eps_linear": float(eps_l),
    }


# ----------------------------------------------------------------- ADV
def estimate_adv(lob: LOBData) -> float:
    """Traded volume over the session, in shares.

    One ticker-day gives one observation, so this is a day's volume rather
    than a true average daily volume; with several days, average the values.
    Visible (type 4) and hidden (type 5) executions both count. LOBSTER logs
    each execution once per resting order, so summing sizes double-counts
    neither side.
    """
    flow = lob.signed_flow()
    return float(flow["size"].sum()) if len(flow) else np.nan


def x_from_adv(adv: float, pct_adv: float) -> float:
    """Order size as a fraction of daily volume (pct_adv in percent)."""
    return float(adv * pct_adv / 100.0)


# ------------------------------------------------- impact via the sqrt law
def eta_from_square_root_law(X: float, adv: float, sigma: float, S0: float,
                             Y: float = 0.5, T: float = 1.0) -> float:
    """Temporary-impact coefficient implied by the empirical square-root law.

        dP / P  =  Y * sigma_rel * sqrt(Q / V)

    with sigma_rel the daily relative volatility, Q the order size, V the daily
    volume and Y ~ 0.5-1 across markets and decades. Converting to the AC
    temporary coefficient at the TWAP rate (cost per share = eta * X / T):

        eta = Y * sigma_rel * sqrt(X / adv) * S0 * T / X

    WHY THIS IS NEEDED. Walking the displayed book measures the mechanical cost
    of consuming visible depth at ONE INSTANT. The cost of working a metaorder
    over hours is dominated instead by information leakage and adverse
    selection, which displayed depth does not price at all. On the LOBSTER
    sample the book-walk estimate came out 37-293x BELOW the square-root law,
    and that single underestimate caused three downstream failures:

      * AC infeasibility: eta_tilde = eta - gamma*tau/2 < 0 for all five names.
        The constraint was not telling us gamma was too big; it was telling us
        eta was too small. With the square-root eta, eta_tilde > 0 everywhere
        at the ORIGINAL fitted gamma, no capping required.
      * Degenerate schedules: kappa*T = 15-31 (dump 78-95% immediately). With
        the square-root eta, kappa*T = 1.0-2.6 at the project's lam = 2e-6.
      * Zero adaptivity value: temporary impact was 0.02-0.13 bps against a
        1.3-2.9 bps spread, so liquidity variation could not matter. With the
        square-root eta, impact is 4.8-7.2 bps and dominates the spread.

    Book-walk eta remains the right estimator for an IMMEDIATE sweep, and its
    fitted EXPONENT is still the informative cross-sectional quantity. Use this
    function for the coefficient when modelling execution over a horizon.
    """
    sigma_rel = sigma / S0
    dP = Y * sigma_rel * np.sqrt(X / adv) * S0   # $/share
    return float(dP * T / X)


# --------------------------------------------------------------- gamma
def estimate_gamma(grid: pd.DataFrame) -> tuple[float, float, float]:
    """Kyle-style regression dMid = gamma * Q + noise.

    Returns (gamma, standard_error, r2). The standard error is essential,
    not decorative: over a single ticker-day the price noise sigma*sqrt(tau)
    is typically two orders of magnitude larger than the impact signal
    gamma*Q, so gamma is often NOT IDENTIFIED and the point estimate is
    meaningless on its own. `calibrate()` flags |gamma| < 2*se.
    """
    # ALIGNMENT: grid["mid"] is the mid at the END of each interval, so
    # diff(mid)[k] is the price change DURING interval k+1 and must be paired
    # with net flow from interval k+1, i.e. net[1:]. Using net[:-1] regresses
    # each change on the PREVIOUS interval's flow, which measures a lagged
    # (near-zero) effect and returned gamma = -1.23e-06 on data whose true
    # value was +2.5e-05.
    q = grid["net_signed_size"].to_numpy(float)[1:]
    dmid = np.diff(grid["mid"].to_numpy(float))
    ok = np.isfinite(q) & np.isfinite(dmid)
    q, dmid = q[ok], dmid[ok]
    n = len(q)
    if n < 10 or np.allclose(q, 0):
        return np.nan, np.nan, np.nan
    sq = float(np.sum(q * q))
    gamma = float(np.sum(q * dmid) / sq)              # no-intercept OLS
    resid = dmid - gamma * q
    s2 = float(np.sum(resid**2) / max(n - 1, 1))      # residual variance
    se = float(np.sqrt(s2 / sq))                      # OLS se for the slope
    ss_tot = float(np.sum((dmid - dmid.mean()) ** 2))
    r2 = float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan
    return gamma, se, r2


# ------------------------------------------------------- OU liquidity fit
def estimate_ou_liquidity(lob: LOBData, bucket_s: float = 300.0,
                          probe_size: float | None = None,
                          snapshots_per_bucket: int = 40,
                          seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Fit an AR(1) to log eta_t estimated in coarse time buckets.

    Returns (theta [1/day], stationary log-std, the log-eta series).
    eta_t is proxied at a single probe size: cost above the half-spread
    divided by the trading rate, which is proportional to eta under the
    linear model and inherits its time variation.
    """
    rng = np.random.default_rng(seed)
    t = lob.time
    mid = lob.mid
    # The probe MUST clear level-1 depth, or it never leaves the touch and the
    # measured "excess cost" is numerical noise -- which produced a nonsense
    # log-std of 13.1 on a book whose true value was 0.30. A fixed default
    # cannot work across a cross-section whose depths differ by ~60x.
    l1 = float(np.median(lob.book["bid_sz_1"].to_numpy()))
    total = float(np.median(np.sum(
        [lob.book[f"bid_sz_{l}"].to_numpy() for l in range(1, lob.n_levels + 1)],
        axis=0)))
    if probe_size is None:
        probe_size = float(min(max(3.0 * l1, 1.0), 0.4 * total))
    elif probe_size < l1:
        print(f"[warn] OU probe size {probe_size:.0f} is below level-1 depth "
              f"{l1:.0f}; the liquidity series will be noise")
    edges = np.arange(t[0], t[-1], bucket_s)
    tau_days = bucket_s / SECONDS_PER_TRADING_DAY
    series = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = np.searchsorted(t, [a, b])
        if hi - lo < 5:
            continue
        pick = rng.choice(np.arange(lo, hi),
                          size=min(snapshots_per_bucket, hi - lo), replace=False)
        costs, halfspreads = [], []
        for i in pick:
            px, sz = lob.bid_ladder(int(i))
            c = walk_book_cost(px, sz, mid[int(i)], probe_size, "sell")
            if np.isfinite(c):
                costs.append(c)
                halfspreads.append(0.5 * lob.spread[int(i)])
        if len(costs) < 3:
            continue
        excess = np.mean(costs) - np.mean(halfspreads)
        # a floor of one hundredth of the half-spread keeps rounding noise out
        # of the log series (log of near-zero explodes the fitted variance)
        if excess > 0.01 * max(np.mean(halfspreads), 1e-9):
            series.append(excess / (probe_size / tau_days))
    if len(series) < 8:
        return np.nan, np.nan, np.array([])

    log_eta = np.log(np.asarray(series))
    x, y = log_eta[:-1], log_eta[1:]
    xc, yc = x - x.mean(), y - y.mean()
    rho = float(np.sum(xc * yc) / np.sum(xc * xc))
    rho = float(np.clip(rho, 1e-6, 0.999999))
    resid = y - (y.mean() + rho * xc)
    innov = resid.std(ddof=1)
    stat_std = float(innov / np.sqrt(max(1 - rho**2, 1e-9)))
    theta = float(-np.log(rho) / tau_days)
    return theta, stat_std, log_eta


# ------------------------------------------------------------ top level
def calibrate(lob: LOBData, interval_s: float = 60.0,
              probe_sizes: np.ndarray | None = None,
              tau_days: float | None = None,
              max_snapshots: int = 3000,
              min_fill: float = 0.95) -> Calibration:
    """Run every estimator on one ticker-day."""
    grid = resample_grid(lob, interval_s)
    sigma = estimate_sigma(grid, interval_s)
    eps = estimate_eps(lob)
    gamma, gamma_se, gamma_r2 = estimate_gamma(grid)

    # Probe sizes must clear top-of-book depth (pitfall (a) above): below it
    # the walk never leaves level 1 and the cost curve is flat.
    l1_depth = float(np.median(lob.book["bid_sz_1"].to_numpy()))
    total_depth = float(np.median(np.sum(
        [lob.book[f"bid_sz_{l}"].to_numpy() for l in range(1, lob.n_levels + 1)],
        axis=0)))
    if probe_sizes is None:
        # lower end clears level-1 depth (pitfall a); upper end stays well
        # inside the displayed book so probes actually fill (pitfall c)
        hi = max(0.5 * total_depth, 3 * l1_depth)
        probe_sizes = np.unique(np.round(
            np.geomspace(2 * l1_depth, hi, 7)).astype(float))
    elif np.min(probe_sizes) < l1_depth:
        print(f"[warn] smallest probe size {np.min(probe_sizes):.0f} is below median "
              f"level-1 depth {l1_depth:.0f}; the flat region will bias the exponent up")
    if tau_days is None:
        tau_days = interval_s / SECONDS_PER_TRADING_DAY

    curve = impact_curve(lob, probe_sizes, max_snapshots=max_snapshots)
    dropped = curve[curve["fill_rate"] < min_fill]
    if len(dropped):
        print(f"[warn] dropping {len(dropped)} probe size(s) with fill rate < "
              f"{min_fill:.0%} (survivorship bias): "
              + ", ".join(f"{r.n_shares:.0f}@{r.fill_rate:.0%}"
                          for r in dropped.itertuples()))
    curve = curve[curve["fill_rate"] >= min_fill]
    if len(curve) < 3:
        raise ValueError(
            f"only {len(curve)} probe sizes fill reliably; the displayed book "
            f"(median total depth {total_depth:.0f}) is too thin for this probe "
            "grid — pass a smaller `probe_sizes` range")
    fit = fit_impact(curve, tau_days, eps_fixed=None)  # free intercept, pitfall (b)
    theta, ou_std, _ = estimate_ou_liquidity(lob)

    if not (gamma > 0):
        print(f"[warn] gamma = {gamma:.3g} is not positive — check the trade-sign "
              "convention (see lobster_io docstring)")
    if np.isfinite(gamma_se) and abs(gamma) < 2 * gamma_se:
        print(f"[warn] gamma NOT IDENTIFIED: {gamma:.3g} +/- {gamma_se:.3g} "
              f"(R^2 = {gamma_r2:.5f}). Over one ticker-day the price noise "
              "swamps the impact signal. Pool several days, or take gamma from "
              "the literature and run a sensitivity analysis instead.")

    return Calibration(
        sigma=sigma, eps=eps, eta=fit["eta_linear"], gamma=gamma,
        impact_power=fit["impact_power"], impact_coef=fit["impact_coef"],
        impact_power_stderr=fit["impact_power_stderr"],
        impact_intercept=fit["eps_from_fit"],
        gamma_se=gamma_se, gamma_r2=gamma_r2,
        probe_min=float(curve["n_shares"].min()),
        probe_max=float(curve["n_shares"].max()),
        adv=estimate_adv(lob), l1_depth=l1_depth,
        theta=theta, ou_log_std=ou_std, n_snapshots=len(lob.book),
        ticker=lob.ticker, date=lob.date,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages")
    ap.add_argument("--orderbook")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--ticker", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="clock-time grid for sigma/gamma, seconds")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic recovery check instead")
    args = ap.parse_args()

    if args.self_test:
        import test_calibration
        raise SystemExit(test_calibration.main())

    if not (args.messages and args.orderbook):
        ap.error("--messages and --orderbook are required (or use --self-test)")

    lob = load_lobster(args.messages, args.orderbook, args.levels,
                       ticker=args.ticker, date=args.date)
    cal = calibrate(lob, interval_s=args.interval)

    print(f"\ncalibration — {cal.ticker or 'ticker'} {cal.date}  "
          f"({cal.n_snapshots:,} snapshots)")
    print("-" * 60)
    print(f"  sigma   {cal.sigma:12.6g}  $/share/sqrt(day)")
    print(f"  eps     {cal.eps:12.6g}  $/share (half-spread)")
    print(f"  eta     {cal.eta:12.6g}  $/share per (share/day)")
    ident = "IDENTIFIED" if abs(cal.gamma) >= 2 * cal.gamma_se else "NOT IDENTIFIED"
    print(f"  gamma   {cal.gamma:12.6g}  +/- {cal.gamma_se:.3g}  "
          f"(R^2 {cal.gamma_r2:.5f})  [{ident}]")
    print(f"  a_fit   {cal.impact_intercept:12.6g}  $/share (walk-curve intercept; "
          f"expect ~eps - tick/2)")
    print(f"  power   {cal.impact_power:12.4f}  +/- {cal.impact_power_stderr:.4f}"
          "   <- 1.0 = linear (AC), 0.5 = square-root")
    print(f"  theta   {cal.theta:12.4f}  1/day   (OU mean reversion)")
    print(f"  ou_std  {cal.ou_log_std:12.4f}          (stationary std of log eta)")
    print("-" * 60)

    lo = cal.impact_power - 2 * cal.impact_power_stderr
    hi = cal.impact_power + 2 * cal.impact_power_stderr
    nearest = "LINEAR (1.0)" if abs(cal.impact_power - 1.0) < abs(cal.impact_power - 0.5) \
        else "SQUARE-ROOT (0.5)"
    inside = "consistent with it" if (lo <= 1.0 <= hi or lo <= 0.5 <= hi) else \
        "but the interval excludes both reference values"
    print(f"impact exponent 95% CI [{lo:.3f}, {hi:.3f}] — closest to {nearest}, {inside}")
    print("  (the CI covers snapshot sampling error only; it does not cover "
          "day-to-day\n   variation, so treat a narrow interval as a lower bound "
          "on uncertainty)")

    print("\nACParams snippet:\n" + cal.to_acparams_snippet(
        S0=float(np.median(lob.mid))))
    with open(args.out, "w") as f:
        json.dump(asdict(cal), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
