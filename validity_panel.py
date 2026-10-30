"""validity_panel.py — are the calibrated parameters usable for the order we
intend to execute?

Calibration produces numbers; it does not certify that those numbers apply to
the problem you want to solve. Two failures showed up on the LOBSTER sample
cross-section, both invisible in the calibration output itself:

1. EXTRAPOLATION. `eta` and the impact exponent are fitted over a probe-size
   range bounded by the displayed book. With X = 1,000,000 shares over N = 50
   intervals, the per-interval size is 20,000 shares — about 0.3x the largest
   fitted probe for INTC/MSFT (fine) but roughly 40x for AMZN/GOOG/AAPL, whose
   books hold ~100 shares at the touch. Applying the fit there extrapolates
   nearly two orders of magnitude past the data.

2. ECONOMIC IMPLAUSIBILITY OF gamma. The Kyle-style regression reports
   gamma "identified" by the statistical test |gamma| >= 2*se, yet implies a
   1M-share order moves AMZN by 27% and GOOG by 21% of their price. The cause
   is simultaneity rather than noise: within an interval, price moves and
   signed flow are mechanically linked, so the coefficient absorbs
   contemporaneous (partly transient) price pressure. Note the synthetic
   tests in test_calibration.py CANNOT catch this, because the generator
   draws trade signs exogenously and so contains no reverse causality --
   a real limit of synthetic validation, and the reason this panel exists.

3. MODEL INFEASIBILITY. Almgren-Chriss needs eta_tilde = eta - gamma*tau/2 > 0;
   otherwise the cost functional stops being convex, kappa is undefined and the
   closed form does not exist. On the LOBSTER cross-section eta_tilde came out
   NEGATIVE for all five names, because the two coefficients are estimated at
   different horizons and measure overlapping things: gamma comes from a
   60-second flow regression and absorbs transient pressure that has not yet
   decayed, while eta comes from an instantaneous book walk. The regression
   gamma is therefore too large relative to the book-walk eta, and the
   decomposition breaks. This is a HARD constraint, not a heuristic, and it
   binds more tightly than the 5%-of-price magnitude test: it caps gamma at
   2*eta/tau, which for INTC is 7.5x below the fitted value even though the
   fitted value implies a perfectly sensible 0.19% permanent move.

This script answers, per ticker: what order size is actually defensible, and
which parameters can be trusted at that size.

Usage
-----
    python validity_panel.py                          # reads calibration_all.csv
    python validity_panel.py --pct-adv 1.0 --n 50     # 1% of ADV, 50 intervals
    python validity_panel.py --x 1000000              # force a fixed size

Writes validity_panel.csv and prints ACParams blocks scaled per ticker.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# A permanent price displacement above this fraction of the price, for the
# order being executed, is treated as economically implausible rather than as
# an estimate. 5% is generous: institutional metaorder studies typically find
# permanent impact of tens of basis points for orders of a few percent of ADV.
GAMMA_PLAUSIBILITY_LIMIT = 0.05

# Prefactor in the square-root law dP/P = Y * sigma_rel * sqrt(Q/V).
# Empirically Y ~ 0.5-1 across markets, asset classes and decades.
SQRT_LAW_Y = 0.5

# Per-interval size beyond this multiple of the largest fitted probe is
# flagged; beyond 10x the impact fit is being extrapolated, not applied.
EXTRAPOLATION_WARN = 3.0
EXTRAPOLATION_FAIL = 10.0


def lam_for_kappa(target_kappa_T: float, sigma: float, eta_tilde: float,
                  T: float = 1.0) -> float:
    """Risk aversion that puts the AC schedule at a chosen urgency.

    kappa^2 = lam * sigma^2 / eta_tilde, and X appears NOWHERE: E[IS] and
    Var[IS] both scale as X^2, so the schedule's SHAPE is size-independent.
    That means order size cannot be used to move between urgency regimes --
    only lam (or the ratio sigma^2/eta) can.

    This matters because the project's lam = 2e-6 was chosen for the toy
    parameters (sigma 0.95, eta 2.5e-6) and gives kappa*T = 0.85, a gentle
    near-TWAP schedule. With CALIBRATED parameters the same lam gives
    kappa*T = 15-31: the AC optimum dumps 78-95% of the order in the first
    10% of the day. That is economically correct -- at 1% of ADV in a deep
    book, impact is negligible against price risk, so trade immediately --
    but it is a degenerate test bed, because a schedule with no interior
    structure leaves nothing for a policy to learn.

    Useful targets: kappa*T ~ 0.5-1 is close to TWAP with mild front-loading;
    2-4 is visibly front-loaded and is where the AC-vs-TWAP gap is largest
    relative to the noise; >10 is effectively an immediate block sale.
    """
    return float(target_kappa_T**2 * eta_tilde / (sigma**2 * T**2))


def panel(df: pd.DataFrame, pct_adv: float | None, x_fixed: float | None,
          n_intervals: int) -> pd.DataFrame:
    out = df.copy()

    if x_fixed is not None:
        out["X"] = float(x_fixed)
        out["pct_adv"] = 100.0 * out["X"] / out["adv"]
    else:
        out["pct_adv"] = float(pct_adv)
        out["X"] = out["adv"] * pct_adv / 100.0

    out["n_per_interval"] = out["X"] / n_intervals
    out["extrap_factor"] = out["n_per_interval"] / out["probe_max"]

    # permanent displacement implied for THIS order, as a fraction of price
    out["perm_impact_$"] = out["gamma"] * out["X"]
    out["perm_impact_pct"] = 100.0 * out["perm_impact_$"] / out["mid_px"]

    # temporary cost per share at the TWAP rate (X shares over one day)
    out["temp_cost_$/sh"] = out["eta"] * out["X"]
    out["temp_cost_bps"] = 1e4 * out["temp_cost_$/sh"] / out["mid_px"]

    # spread cost is unavoidable and sets a floor for comparison
    out["spread_cost_bps"] = 1e4 * out["eps"] / out["mid_px"]

    # AC feasibility: eta_tilde = eta - gamma*tau/2 must be positive
    tau = 1.0 / n_intervals
    out["gamma_max_ac"] = 2.0 * out["eta"] / tau
    out["eta_tilde"] = out["eta"] - 0.5 * out["gamma"] * tau
    out["ac_feasible"] = out["eta_tilde"] > 0

    # square-root-law eta: the coefficient appropriate for execution over a
    # horizon, as opposed to the book-walk value which prices an instant sweep
    sigma_rel = out["sigma"] / out["mid_px"]
    out["eta_sqrt"] = (SQRT_LAW_Y * sigma_rel * np.sqrt(out["X"] / out["adv"])
                       * out["mid_px"] / out["X"])
    out["eta_ratio"] = out["eta_sqrt"] / out["eta"]
    out["eta_tilde_sqrt"] = out["eta_sqrt"] - 0.5 * out["gamma"] * tau
    out["ac_feasible_sqrt"] = out["eta_tilde_sqrt"] > 0
    out["kappaT_sqrt"] = np.sqrt(
        2e-6 * out["sigma"]**2 / out["eta_tilde_sqrt"].clip(lower=1e-30))

    out["eta_valid"] = out["extrap_factor"] <= EXTRAPOLATION_FAIL
    # Plausibility is two-sided. A NEGATIVE gamma says trading pushes the price
    # in your favour, which is not a small estimate — it is a failed one, and
    # the magnitude test alone would wave it through.
    out["gamma_plausible"] = (
        (out["perm_impact_pct"] > 0)
        & (out["perm_impact_pct"] <= 100 * GAMMA_PLAUSIBILITY_LIMIT)
    )
    out["gamma_sign_ok"] = out["gamma"] > 0
    return out


def verdict(r) -> str:
    parts = []
    if not r.eta_valid:
        parts.append("eta EXTRAPOLATED — shrink X or use more book levels")
    if not r.gamma_sign_ok:
        parts.append("gamma NEGATIVE — not identified; sensitivity axis only")
    elif not r.gamma_plausible:
        parts.append("gamma IMPLAUSIBLE (too large) — sensitivity axis only")
    if not r.ac_feasible:
        parts.append("AC INFEASIBLE (eta_tilde < 0) — gamma must be capped")
    return "usable" if not parts else "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="calibration_all.csv")
    ap.add_argument("--pct-adv", type=float, default=1.0,
                    help="order size as %% of the session's traded volume")
    ap.add_argument("--x", type=float, default=None,
                    help="force a fixed share count instead of ADV scaling")
    ap.add_argument("--n", type=int, default=50, help="number of intervals")
    ap.add_argument("--out", default="validity_panel.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.calibration)
    need = {"ticker", "mid_px", "eta", "gamma", "adv", "probe_max"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(
            f"{args.calibration} is missing {sorted(missing)} — regenerate it with "
            "the current calibrate_all.py (older runs predate ADV/probe recording)."
        )

    p = panel(df, args.pct_adv, args.x, args.n).sort_values("mid_px")

    mode = (f"X = {args.x:,.0f} shares (fixed)" if args.x
            else f"X = {args.pct_adv:.2f}% of session volume")
    print("=" * 104)
    print(f"VALIDITY PANEL — {mode}, N = {args.n} intervals")
    print("=" * 104)
    show = p.copy()
    for c, d in [("adv", 0), ("X", 0), ("n_per_interval", 0), ("probe_max", 0),
                 ("extrap_factor", 1), ("perm_impact_pct", 2),
                 ("temp_cost_bps", 2), ("spread_cost_bps", 2), ("pct_adv", 2)]:
        show[c] = show[c].round(d)
    show["gamma_max_ac"] = show["gamma_max_ac"].map(lambda v: f"{v:.3g}")
    show["gamma_disp"] = p["gamma"].map(lambda v: f"{v:.3g}")
    print(show[["ticker", "adv", "pct_adv", "X", "n_per_interval", "probe_max",
                "extrap_factor", "temp_cost_bps", "spread_cost_bps",
                "perm_impact_pct", "gamma_disp", "gamma_max_ac",
                "ac_feasible"]].to_string(index=False))

    print("\n--- per-ticker verdict ---")
    for r in p.itertuples():
        print(f"  {r.ticker:<5} {verdict(r)}")
        if not r.eta_valid:
            x_ok = r.probe_max * EXTRAPOLATION_FAIL * args.n
            print(f"        per-interval {r.n_per_interval:,.0f} vs largest fitted "
                  f"probe {r.probe_max:,.0f} ({r.extrap_factor:.1f}x). "
                  f"X <= {x_ok:,.0f} keeps the fit applicable "
                  f"({100*x_ok/r.adv:.2f}% of volume).")
        if not r.ac_feasible:
            print(f"        eta_tilde = {r.eta_tilde:.3g} < 0: AC's closed form does "
                  f"not exist. gamma must satisfy gamma < 2*eta/tau = "
                  f"{r.gamma_max_ac:.3g} (fitted {r.gamma:.3g}, "
                  f"{r.gamma / r.gamma_max_ac:.1f}x too large).")
        if not r.gamma_plausible:
            g_max = GAMMA_PLAUSIBILITY_LIMIT * r.mid_px / r.X
            if not r.gamma_sign_ok:
                print(f"        fitted gamma is negative ({r.gamma:.3g}): selling "
                      "would raise the price. Use the ceiling "
                      f"{g_max:.3g} as the upper end of a sensitivity range "
                      "starting at 0.")
            else:
                print(f"        gamma implies {r.perm_impact_pct:.1f}% permanent price "
                      f"move for this order. A plausible ceiling is "
                      f"{g_max:.3g} (vs fitted {r.gamma:.3g}).")

    print("\n--- notes ---")
    print("* extrap_factor = per-interval size / largest probe size actually fitted.")
    print(f"  <= {EXTRAPOLATION_WARN:.0f}x is comfortable; > {EXTRAPOLATION_FAIL:.0f}x means "
          "the impact curve is being extrapolated, not applied.")
    print("* eta from walking the DISPLAYED book prices an IMMEDIATE SWEEP, which")
    print("  badly understates the cost of working an order over hours: leakage and")
    print("  adverse selection are not in displayed depth. On this cross-section it")
    print("  came out 37-293x below the square-root law. The eta_sqrt column uses")
    print("  the law instead; it is what the week-7 environments should use.")
    print("* eta_tilde = eta - gamma*tau/2 must exceed 0 or AC has no closed form.")
    print("  This binds at gamma < 2*eta/tau and is usually TIGHTER than the")
    print("  magnitude test, because the two coefficients are fitted at different")
    print("  horizons (60s flow regression vs instantaneous book walk).")
    print("* gamma_plausible only bounds magnitude; it does not fix the simultaneity")
    print("  bias. Treat gamma as a sensitivity axis with the ceiling above as its")
    print("  upper end, and re-run the week-5 comparisons across that range.")

    p.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    print("\n--- square-root-law eta (use THIS for week-7 environments) ---")
    print(f"{'tkr':<6} {'eta (book walk)':>16} {'eta (sqrt law)':>15} {'ratio':>7} "
          f"{'eta_tilde>0':>12} {'kappa*T @2e-6':>14}")
    for r in p.itertuples():
        print(f"{r.ticker:<6} {r.eta:16.4g} {r.eta_sqrt:15.4g} {r.eta_ratio:6.0f}x "
              f"{str(r.ac_feasible_sqrt):>12} {r.kappaT_sqrt:14.2f}")
    print("  -> with the sqrt-law eta, AC is feasible at the FITTED gamma (no cap)")
    print("     and the schedule is non-degenerate at the project's lam = 2e-6.")

    print("\n--- risk aversion needed for a non-degenerate schedule ---")
    print("kappa*T is the urgency; X does not enter it, so only lam moves the regime.")
    print(f"{'tkr':<6} {'kappa*T @ lam=2e-6':>19} {'lam for kT=1':>14} {'lam for kT=3':>14}")
    for r in p.itertuples():
        ceiling = min(GAMMA_PLAUSIBILITY_LIMIT * r.mid_px / r.X, 0.5 * r.gamma_max_ac)
        g = min(r.gamma, ceiling) if r.gamma > 0 else 0.5 * ceiling
        et = r.eta - 0.5 * g / args.n
        if et <= 0:
            continue
        kT = np.sqrt(2e-6 * r.sigma**2 / et)
        print(f"{r.ticker:<6} {kT:19.1f} "
              f"{lam_for_kappa(1.0, r.sigma, et):14.3g} "
              f"{lam_for_kappa(3.0, r.sigma, et):14.3g}")
    print("  -> the project's lam = 2e-6 is ~100-1000x too large for calibrated")
    print("     parameters; week 7 should sweep lam by target kappa*T, not fix it.")

    print("\n--- RECOMMENDED ACParams for week 7 (sqrt-law eta) ---")
    for r in p.itertuples():
        if not r.ac_feasible_sqrt:
            continue
        print(f"\n# {r.ticker}: X = {r.X:,.0f} ({r.pct_adv:.2f}% of volume), "
              f"kappa*T = {r.kappaT_sqrt:.2f}")
        print(f"ACParams(X={r.X:,.0f}".replace(",", "_") +
              f", S0={r.mid_px:.2f}, T=1.0, N={args.n},")
        print(f"         sigma={r.sigma:.6g}, gamma={r.gamma:.6g},")
        print(f"         eta={r.eta_sqrt:.6g}, eps={r.eps:.6g}, lam=2e-6)")

    print("\n--- ACParams with book-walk eta (NOT recommended; kept for contrast) ---")
    for r in p.itertuples():
        # two ceilings bind: economic plausibility, and AC feasibility
        # (eta_tilde > 0). Use the tighter, with a margin so eta_tilde is not
        # merely positive but not vanishing.
        ceiling = min(GAMMA_PLAUSIBILITY_LIMIT * r.mid_px / r.X,
                      0.5 * r.gamma_max_ac)
        g_use = min(r.gamma, ceiling) if r.gamma > 0 else 0.5 * ceiling
        print(f"\n# {r.ticker}: X = {r.X:,.0f} ({r.pct_adv:.2f}% of volume), "
              f"extrapolation {r.extrap_factor:.1f}x")
        print(f"ACParams(X={r.X:,.0f}".replace(",", "_") +
              f", S0={r.mid_px:.2f}, T=1.0, N={args.n},")
        et_use = r.eta - 0.5 * g_use / args.n
        lam_use = (lam_for_kappa(2.0, r.sigma, et_use) if et_use > 0 else 2e-6)
        print(f"         sigma={r.sigma:.6g}, gamma={g_use:.6g},")
        print(f"         eta={r.eta:.6g}, eps={r.eps:.6g}, lam={lam_use:.6g})")
        print(f"# lam set for kappa*T = 2 (non-degenerate); "
              f"lam = 2e-6 would give kappa*T = "
              f"{np.sqrt(2e-6 * r.sigma**2 / et_use) if et_use > 0 else float('nan'):.0f}")
        if r.gamma <= 0:
            print(f"# gamma PLACEHOLDER (fitted {r.gamma:.3g} < 0, not identified); "
                  f"sweep 0 .. {ceiling:.3g}")
        elif g_use < r.gamma:
            why = ("AC feasibility (eta_tilde > 0)"
                   if 0.5 * r.gamma_max_ac <= GAMMA_PLAUSIBILITY_LIMIT * r.mid_px / r.X
                   else "economic plausibility")
            print(f"# gamma capped by {why}: fitted {r.gamma:.3g} -> {g_use:.3g}; "
                  f"sweep 0 .. {ceiling:.3g}")


if __name__ == "__main__":
    main()
