"""test_calibration.py — validate every estimator against known ground truth.

The discipline that has carried this project: never trust a number on real
data that you have not first recovered on synthetic data where the answer is
known. Here we generate a synthetic LOBSTER ticker-day from an explicit
model, run the full calibration pipeline, and check each estimate lands on
its true value.

Synthetic generator
-------------------
Midprice: arithmetic random walk with known sigma, plus permanent impact
    mid_{k+1} = mid_k + sigma_step * xi + gamma_true * Q_k
where Q_k is the net signed size traded in interval k.

Book: both sides are ladders of `depth_t` shares at every tick above/below
the touch, with a known half-spread. Walking n shares into that ladder costs

    cost_per_share(n) = half_spread + (tick / (2 * depth)) * n    (approx)

so the TRUE linear impact coefficient is
    eta_true = tick * tau_days / (2 * depth)       [$/share per (share/day)]
because the estimator fits against the RATE n/tau, not n.

Liquidity: depth_t follows a lognormal OU with known theta and stationary
std, which induces a lognormal-OU eta_t with the SAME theta and std — the
ground truth for the OU fit.

Square-root arm: a second generator builds a ladder whose cumulative depth
grows quadratically with distance from the touch, which makes the walk cost
grow as sqrt(size) — the test that `fit_impact` can actually distinguish
the two regimes rather than always reporting ~1.

Tests
-----
 1. eps recovery
 2. sigma recovery
 3. gamma recovery (and a deliberate sign-flip: the wrong convention must
    produce gamma < 0, proving the test would catch it)
 4. eta recovery via walking the book
 5. impact exponent ~ 1.0 on a linear book
 6. impact exponent ~ 0.5 on a square-root book
 7. OU theta and log-std recovery
 8. walk_book_cost unit checks (exact arithmetic on a hand-built ladder)

Run:  python test_calibration.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from lobster_io import LOBData, resample_grid
from calibrate import (
    SECONDS_PER_TRADING_DAY, estimate_sigma, estimate_eps, estimate_gamma,
    impact_curve, fit_impact, estimate_ou_liquidity, walk_book_cost,
)

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def within(est, true, tol_frac, name, unit=""):
    rel = abs(est - true) / abs(true) if true != 0 else abs(est)
    check(name, rel <= tol_frac,
          f"est {est:.6g}{unit} vs true {true:.6g}{unit} (rel err {rel:.3%}, tol {tol_frac:.0%})")
    return rel


# ----------------------------------------------------------- generator
def make_synthetic(
    n_events: int = 60_000,
    n_levels: int = 10,
    tick: float = 0.01,
    half_spread_ticks: float = 0.5,
    depth: float = 500.0,
    sigma_day: float = 0.95,
    gamma_true: float = 2.5e-7,
    trade_rate: float = 0.25,
    mean_trade_size: float = 200.0,
    book_shape: str = "linear",
    ou_theta: float | None = None,
    ou_log_std: float = 0.0,
    S0: float = 50.0,
    seed: int = 0,
) -> tuple[LOBData, dict]:
    """Build a synthetic LOBSTER ticker-day with known parameters.

    book_shape:
      'linear' — constant `depth` shares per tick => walk cost linear in size
      'sqrt'   — depth grows linearly with level index => cumulative depth is
                 quadratic in distance, so walk cost grows as sqrt(size)
    """
    rng = np.random.default_rng(seed)
    dt = SECONDS_PER_TRADING_DAY / n_events
    t = 34200.0 + np.arange(n_events) * dt
    sigma_step = sigma_day * np.sqrt(dt / SECONDS_PER_TRADING_DAY)

    # --- trades and permanent impact
    is_trade = rng.random(n_events) < trade_rate
    sizes = np.where(
        is_trade,
        np.maximum(1.0, np.round(rng.exponential(mean_trade_size, n_events))),
        np.maximum(1.0, np.round(rng.exponential(mean_trade_size, n_events))),
    )
    aggressor = rng.choice([-1.0, 1.0], size=n_events)
    signed = np.where(is_trade, aggressor * sizes, 0.0)

    noise = rng.normal(0.0, sigma_step, n_events)
    mid = S0 + np.cumsum(noise + gamma_true * signed)

    # --- time-varying depth (OU) or constant
    if ou_theta is not None and ou_log_std > 0:
        rho = np.exp(-ou_theta * dt / SECONDS_PER_TRADING_DAY)
        z = np.empty(n_events)
        z[0] = rng.standard_normal()
        innov = np.sqrt(1 - rho**2)
        for k in range(1, n_events):
            z[k] = rho * z[k - 1] + innov * rng.standard_normal()
        depth_t = depth * np.exp(ou_log_std * z - 0.5 * ou_log_std**2)
    else:
        depth_t = np.full(n_events, float(depth))

    # --- build the ladders
    hs = half_spread_ticks * tick
    cols = {}
    for l in range(1, n_levels + 1):
        cols[f"ask_px_{l}"] = mid + hs + (l - 1) * tick
        cols[f"bid_px_{l}"] = mid - hs - (l - 1) * tick
        if book_shape == "linear":
            lvl_depth = depth_t
        elif book_shape == "sqrt":
            # depth grows linearly with level -> cumulative quadratic ->
            # price displacement ~ sqrt(size)
            lvl_depth = depth_t * (2 * l - 1) / 3.0
        else:
            raise ValueError(book_shape)
        cols[f"ask_sz_{l}"] = lvl_depth
        cols[f"bid_sz_{l}"] = lvl_depth
    book = pd.DataFrame(cols)
    book = book[[c for l in range(1, n_levels + 1)
                 for c in (f"ask_px_{l}", f"ask_sz_{l}", f"bid_px_{l}", f"bid_sz_{l}")]]

    # --- messages; LOBSTER direction for a trade is the RESTING side,
    #     i.e. the negation of the aggressor sign
    msg = pd.DataFrame({
        "time": t,
        "type": np.where(is_trade, 4, 1),
        "order_id": np.arange(n_events),
        "size": sizes,
        "price": np.where(aggressor > 0, mid + hs, mid - hs),
        "direction": np.where(is_trade, -aggressor, rng.choice([-1.0, 1.0], n_events)),
    })

    truth = {
        "sigma": sigma_day,
        "eps": hs,
        "gamma": gamma_true,
        "tick": tick,
        "depth": depth,
        "ou_theta": ou_theta,
        "ou_log_std": ou_log_std,
        "book_shape": book_shape,
        "dt": dt,
    }
    lob = LOBData(messages=msg, book=book, n_levels=n_levels,
                  ticker="SYNTH", date="2012-06-21")
    return lob, truth


def eta_true_for(truth: dict, tau_days: float) -> float:
    """Linear-book ground truth for eta, in $/share per (share/day).

    Walking n shares consumes n/depth ticks of ladder; the average price
    concession is about half the final one, so
        cost_per_share ~ half_spread + (tick / (2*depth)) * n
    The estimator fits against rate = n/tau, so eta = tick*tau/(2*depth).
    """
    return truth["tick"] * tau_days / (2.0 * truth["depth"])


# --------------------------------------------------------------- tests
def test_walk_book_unit():
    px = np.array([10.00, 9.99, 9.98])
    sz = np.array([100.0, 100.0, 100.0])
    mid = 10.005
    c = walk_book_cost(px, sz, mid, 100.0, "sell")
    check("walk_book: single level exact", abs(c - (mid - 10.00)) < 1e-12,
          f"{c:.6f} == {mid - 10.00:.6f}")
    c = walk_book_cost(px, sz, mid, 200.0, "sell")
    expect = mid - (10.00 * 100 + 9.99 * 100) / 200
    check("walk_book: two levels exact", abs(c - expect) < 1e-12,
          f"{c:.6f} == {expect:.6f}")
    c = walk_book_cost(px, sz, mid, 150.0, "sell")
    expect = mid - (10.00 * 100 + 9.99 * 50) / 150
    check("walk_book: partial level exact", abs(c - expect) < 1e-12)
    c = walk_book_cost(px, sz, mid, 10_000.0, "sell")
    check("walk_book: returns nan when ladder too thin", np.isnan(c))


def test_core_recovery():
    lob, truth = make_synthetic(seed=1)
    interval_s = 60.0
    grid = resample_grid(lob, interval_s)

    within(estimate_eps(lob), truth["eps"], 0.02, "eps recovery", " $/sh")
    within(estimate_sigma(grid, interval_s), truth["sigma"], 0.15,
           "sigma recovery", " $/sh/sqrt(d)")

    # gamma at REALISTIC signal strength is not identified over one day; the
    # estimator must say so rather than returning a confident wrong number.
    g, se, r2 = estimate_gamma(grid)
    check("gamma standard error is reported", np.isfinite(se), f"se = {se:.3g}")
    check("weak-signal gamma is flagged as unidentified (|g| < 2se)",
          abs(g) < 2 * se,
          f"gamma {g:.3g} +/- {se:.3g}, R^2 {r2:.5f} — one ticker-day is not enough")

    # With a strong enough signal the SAME estimator must recover gamma and
    # the true value must lie inside the reported confidence interval.
    lob_s, truth_s = make_synthetic(n_events=120_000, gamma_true=2.5e-5,
                                    sigma_day=0.30, trade_rate=0.6, seed=21)
    gs, ses, r2s = estimate_gamma(resample_grid(lob_s, interval_s))
    within(gs, truth_s["gamma"], 0.20, "gamma recovery (strong signal)")
    check("true gamma inside 2-sigma CI", abs(gs - truth_s["gamma"]) < 2 * ses,
          f"{gs:.4g} +/- {ses:.3g} contains {truth_s['gamma']:.4g}")
    check("strong-signal gamma is positive (sign convention)", gs > 0)

    # deliberate sign flip: the WRONG convention must give gamma < 0
    bad = lob_s.messages.copy()
    bad["direction"] = -bad["direction"]
    lob_bad = LOBData(messages=bad, book=lob_s.book, n_levels=lob_s.n_levels)
    g_bad, _, _ = estimate_gamma(resample_grid(lob_bad, interval_s))
    check("flipped trade sign is detectable (gamma < 0)", g_bad < 0,
          f"gamma_flipped = {g_bad:.3g}")


def test_fill_rate_filter():
    """Probes that exceed the displayed book must be dropped, not averaged.

    With OU-varying depth, large probes fill only on unusually deep books, so
    their measured cost is selected from the cheapest snapshots. Unfiltered,
    the fitted exponent collapses toward 0 on a known-LINEAR book; filtering
    on fill rate restores it.
    """
    lob, truth = make_synthetic(n_events=40_000, n_levels=10, book_shape="linear",
                                ou_theta=12.0, ou_log_std=0.4, seed=11)
    tau_days = 60.0 / SECONDS_PER_TRADING_DAY
    l1 = float(np.median(lob.book["bid_sz_1"].to_numpy()))
    probes = np.round(l1 * np.array([2, 3, 5, 8, 12, 18, 25, 35])).astype(float)
    curve = impact_curve(lob, probes, max_snapshots=400, seed=1)

    p_unfiltered = fit_impact(curve, tau_days, eps_fixed=None)["impact_power"]
    kept = curve[curve["fill_rate"] >= 0.95]
    p_filtered = fit_impact(kept, tau_days, eps_fixed=None)["impact_power"]

    check("fill_rate is reported per probe size", "fill_rate" in curve.columns)
    check("unfiltered curve collapses the exponent (the bug)", p_unfiltered < 0.6,
          f"power unfiltered = {p_unfiltered:.3f} on a LINEAR book")
    check("fill-rate filter restores the exponent", abs(p_filtered - 1.0) < 0.2,
          f"power filtered = {p_filtered:.3f} (kept {len(kept)}/{len(curve)} sizes)")


def test_impact_linear():
    lob, truth = make_synthetic(book_shape="linear", n_levels=20, seed=2)
    tau_days = 60.0 / SECONDS_PER_TRADING_DAY
    # probe sizes must clear level-1 depth (pitfall (a)); free intercept (b)
    d = truth["depth"]
    sizes = d * np.array([2., 3., 5., 8., 12., 16.])
    curve = impact_curve(lob, sizes, max_snapshots=600, seed=3)
    fit = fit_impact(curve, tau_days, eps_fixed=None)

    within(fit["eta_linear"], eta_true_for(truth, tau_days), 0.20,
           "eta recovery (walk the book)")
    p, se = fit["impact_power"], fit["impact_power_stderr"]
    check("linear book -> exponent ~ 1.0", abs(p - 1.0) < 0.15,
          f"power = {p:.3f} +/- {se:.3f}")
    # the fitted intercept should sit near eps - tick/2, NOT at eps
    a_expect = truth["eps"] - truth["tick"] / 2
    check("walk-curve intercept ~ eps - tick/2 (not eps)",
          abs(fit["eps_from_fit"] - a_expect) < 0.3 * truth["tick"],
          f"a_fit {fit['eps_from_fit']:.6f} vs expected {a_expect:.6f}, eps {truth['eps']:.6f}")


def test_probe_size_pitfall():
    """Including sub-depth probe sizes must visibly bias the exponent up.

    This guards the guidance itself: if a future refactor silently allows
    small probes, this test fails loudly rather than the bias going unnoticed.
    """
    lob, truth = make_synthetic(book_shape="linear", n_levels=20, seed=2)
    tau_days = 60.0 / SECONDS_PER_TRADING_DAY
    d = truth["depth"]
    # same span, same count; only the lower end differs
    good = d * np.array([2., 3., 4., 6., 8., 12.])
    bad = d * np.array([0.2, 0.4, 0.8, 6., 8., 12.])
    p_good = fit_impact(impact_curve(lob, good, max_snapshots=400, seed=3),
                        tau_days, eps_fixed=None)["impact_power"]
    p_bad = fit_impact(impact_curve(lob, bad, max_snapshots=400, seed=3),
                       tau_days, eps_fixed=None)["impact_power"]
    check("sub-depth probes bias the exponent upward (guard)",
          p_bad > p_good + 0.08,
          f"power {p_bad:.3f} with flat region vs {p_good:.3f} with probes "
          f"clearing depth (true 1.0)")


def test_impact_sqrt():
    lob, truth = make_synthetic(book_shape="sqrt", n_levels=20, depth=400.0, seed=4)
    tau_days = 60.0 / SECONDS_PER_TRADING_DAY
    l1 = truth["depth"] / 3.0  # level-1 depth for the sqrt ladder
    sizes = l1 * np.array([4., 8., 16., 28., 44., 64.])
    curve = impact_curve(lob, sizes, max_snapshots=600, seed=5)
    fit = fit_impact(curve, tau_days, eps_fixed=None)
    p, se = fit["impact_power"], fit["impact_power_stderr"]
    check("sqrt book -> exponent clearly below 1", p < 0.8,
          f"power = {p:.3f} +/- {se:.3f}")
    check("sqrt book -> exponent near 0.5", abs(p - 0.5) < 0.2,
          f"power = {p:.3f}")


def test_ou_probe_sizing_deep_book():
    """A deep book must not break the OU fit.

    estimate_ou_liquidity once used a fixed probe of 2000 shares. On a book
    with 2500 shares at the touch the probe never left level 1, the measured
    "excess cost" was numerical noise, and the fitted log-std came out at
    13.09 against a true 0.30. The probe is now derived from book depth.
    """
    lob, truth = make_synthetic(n_events=30_000, n_levels=10, depth=2500.0,
                                ou_theta=10.0, ou_log_std=0.30, seed=3)
    theta, std, series = estimate_ou_liquidity(lob, bucket_s=300.0, seed=7)
    check("deep book: OU log-std stays sane", 0.0 < std < 3.0,
          f"log-std {std:.3f} vs true {truth['ou_log_std']:.2f} "
          "(the bug produced 13.09)")
    within(std, truth["ou_log_std"], 0.60, "deep book: OU log-std recovery")


def test_ou_recovery():
    theta_true, std_true = 12.0, 0.45
    lob, truth = make_synthetic(n_events=90_000, ou_theta=theta_true,
                                ou_log_std=std_true, n_levels=20, seed=6)
    theta, std, series = estimate_ou_liquidity(lob, bucket_s=120.0,
                                               probe_size=1500.0, seed=7)
    check("OU fit produced a series", len(series) >= 8, f"{len(series)} buckets")
    within(std, std_true, 0.45, "OU stationary log-std recovery")
    check("OU theta recovered to right order", 0.25 * theta_true <= theta <= 4 * theta_true,
          f"theta est {theta:.2f} vs true {theta_true:.2f} (1/day)")


def main():
    print("--- unit: walking the book ---")
    test_walk_book_unit()
    print("--- recovery: eps, sigma, gamma (+ sign-flip guard) ---")
    test_core_recovery()
    print("--- recovery: linear book ---")
    test_impact_linear()
    print("--- guard: fill-rate / survivorship ---")
    test_fill_rate_filter()
    print("--- guard: probe-size pitfall ---")
    test_probe_size_pitfall()
    print("--- discrimination: square-root book ---")
    test_impact_sqrt()
    print("--- recovery: OU liquidity ---")
    test_ou_probe_sizing_deep_book()
    test_ou_recovery()
    print("\nAll calibration checks passed." if OK else "\nSOME CHECKS FAILED.")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
