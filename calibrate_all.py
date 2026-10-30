"""calibrate_all.py — calibrate every LOBSTER sample ticker and compare.

The five free LOBSTER samples (2012-06-21: AAPL, AMZN, GOOG, INTC, MSFT) span
the microstructure spectrum, from a tick-constrained low-priced name with deep
queues (INTC, ~$27) to high-priced names with wide spreads and thin books
(AMZN/GOOG, ~$220-570). Calibrating all five turns week 6 from "fit the model
to one stock" into a cross-sectional result:

  - does the fitted impact EXPONENT differ systematically with price/tick
    regime? (i.e. is the linear AC model defensible for some names and not
    others — the empirical version of week 5's P1 question)
  - how much does the OU liquidity variation (log-std) differ across names?
    That parameter is what makes adaptivity valuable in week 5, so its real
    magnitude bounds how large the RL gain can plausibly be.

Usage — point it at the directory holding the unzipped sample files:

    python calibrate_all.py --dir .
    python calibrate_all.py --dir ~/lobster --levels 10 --interval 60

It auto-discovers *_message_<L>.csv / *_orderbook_<L>.csv pairs, so you do not
have to type the long timestamped filenames. Writes calibration_all.csv and
prints a comparison table plus per-ticker ACParams snippets.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from lobster_io import load_lobster
from calibrate import calibrate


def discover(directory: str, levels: int) -> list[tuple[str, str, str, str]]:
    """Find (ticker, date, message_path, orderbook_path) tuples."""
    out = []
    pat = os.path.join(directory, f"*_message_{levels}.csv")
    for msg in sorted(glob.glob(pat)):
        book = msg.replace(f"_message_{levels}.csv", f"_orderbook_{levels}.csv")
        if not os.path.exists(book):
            print(f"[skip] no orderbook file matching {os.path.basename(msg)}")
            continue
        base = os.path.basename(msg)
        m = re.match(r"([A-Z.]+)_(\d{4}-\d{2}-\d{2})_", base)
        ticker = m.group(1) if m else base.split("_")[0]
        date = m.group(2) if m else ""
        out.append((ticker, date, msg, book))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory with unzipped LOBSTER files")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--max-snapshots", type=int, default=3000)
    ap.add_argument("--out", default="calibration_all.csv")
    args = ap.parse_args()

    jobs = discover(args.dir, args.levels)
    if not jobs:
        raise SystemExit(
            f"No *_message_{args.levels}.csv files found in {args.dir!r}. "
            "Unzip the LOBSTER samples there, or pass --levels to match your files."
        )
    print(f"found {len(jobs)} ticker-day(s) at level {args.levels}\n")

    rows = []
    for ticker, date, msg, book in jobs:
        print(f"=== {ticker} {date} " + "=" * 40)
        try:
            lob = load_lobster(msg, book, args.levels, ticker=ticker, date=date)
            cal = calibrate(lob, interval_s=args.interval,
                            max_snapshots=args.max_snapshots)
        except Exception as e:  # a thin book can legitimately fail the fill filter
            print(f"[FAILED] {ticker}: {e}\n")
            continue
        mid = float(np.median(lob.mid))
        l1 = float(np.median(lob.book["bid_sz_1"].to_numpy()))
        rows.append({
            "ticker": ticker, "date": date, "mid_px": mid,
            "spread_bps": 1e4 * (2 * cal.eps) / mid,
            "l1_depth": l1, "adv": cal.adv,
            "probe_min": cal.probe_min, "probe_max": cal.probe_max,
            "sigma": cal.sigma, "eps": cal.eps, "eta": cal.eta,
            "power": cal.impact_power, "power_se": cal.impact_power_stderr,
            "gamma": cal.gamma, "gamma_se": cal.gamma_se,
            "gamma_ident": abs(cal.gamma) >= 2 * cal.gamma_se,
            "theta": cal.theta, "ou_log_std": cal.ou_log_std,
            "snapshots": cal.n_snapshots,
        })
        print()

    if not rows:
        raise SystemExit("No ticker calibrated successfully.")
    df = pd.DataFrame(rows).sort_values("mid_px")

    print("\n" + "=" * 100)
    print("CROSS-SECTION (sorted by price level)")
    print("=" * 100)
    show = df.copy()
    show["spread_bps"] = show["spread_bps"].round(2)
    show["mid_px"] = show["mid_px"].round(2)
    show["l1_depth"] = show["l1_depth"].round(0)
    show["power"] = show["power"].round(3)
    show["power_se"] = show["power_se"].round(3)
    show["sigma"] = show["sigma"].round(4)
    show["eps"] = show["eps"].round(4)
    show["ou_log_std"] = show["ou_log_std"].round(3)
    show["theta"] = show["theta"].round(2)
    print(show[["ticker", "mid_px", "spread_bps", "l1_depth", "sigma", "eps",
                "eta", "power", "power_se", "theta", "ou_log_std",
                "gamma_ident"]].to_string(index=False))

    print("\n--- reading the exponent column ---")
    for r in df.itertuples():
        lo, hi = r.power - 2 * r.power_se, r.power + 2 * r.power_se
        near = "linear" if abs(r.power - 1) < abs(r.power - 0.5) else "sqrt"
        covers = "covers 1.0" if lo <= 1.0 <= hi else (
            "covers 0.5" if lo <= 0.5 <= hi else "covers neither")
        print(f"  {r.ticker:<5} power {r.power:.3f} [{lo:.3f}, {hi:.3f}] "
              f"-> closest to {near}, {covers}")

    if df["power"].notna().sum() >= 3:
        c = df[["mid_px", "power"]].corr().iloc[0, 1]
        print(f"\ncorrelation(price level, exponent) = {c:+.3f} across "
              f"{len(df)} names — a strong negative/positive value suggests the "
              "impact model that fits depends on the tick regime, which is a\n"
              "cross-sectional finding worth reporting (n is small; treat as "
              "indicative, not a test).")

    print("\n--- order-size validity (X = 1M shares, N = 50) ---")
    for r in df.itertuples():
        nk = 1_000_000 / 50
        f = nk / r.probe_max if r.probe_max > 0 else np.nan
        pct = 100 * 1_000_000 / r.adv if r.adv > 0 else np.nan
        flag = "OK" if f <= 10 else "EXTRAPOLATED"
        print(f"  {r.ticker:<5} session volume {r.adv:>12,.0f} sh; 1M = {pct:6.1f}% of it; "
              f"per-interval {nk:,.0f} vs top probe {r.probe_max:,.0f} "
              f"({f:5.1f}x) [{flag}]")
    print("  -> run validity_panel.py to scale X per ticker and check gamma plausibility")

    print(f"\nOU log-std across names: "
          f"{df['ou_log_std'].min():.3f} to {df['ou_log_std'].max():.3f} "
          f"(week-5 experiments used 0.400)")
    print("This is the parameter that makes adaptivity valuable: if the real "
          "values are\nsmaller than 0.400, the week-5 RL gain over AC should be "
          "scaled down accordingly.")

    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    print("\n--- per-ticker ACParams ---")
    for r in df.itertuples():
        print(f"\n# {r.ticker} {r.date}")
        print(f"ACParams(X=1_000_000, S0={r.mid_px:.2f}, T=1.0, N=50,")
        print(f"         sigma={r.sigma:.6g}, gamma={r.gamma:.6g},")
        print(f"         eta={r.eta:.6g}, eps={r.eps:.6g}, lam=2e-6)")
        if not r.gamma_ident:
            print(f"# NOTE: gamma not identified ({r.gamma:.3g} +/- {r.gamma_se:.3g});"
                  " treat as a sensitivity axis, not an estimate")


if __name__ == "__main__":
    main()
