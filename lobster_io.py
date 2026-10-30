"""lobster_io.py — load and normalise LOBSTER limit-order-book data.

LOBSTER ships two CSVs per ticker/day, with matching rows:

message file  _message_<level>.csv
    Time, Type, OrderID, Size, Price, Direction
      Time      seconds after midnight (float, to nanoseconds)
      Type      1 new limit order, 2 partial cancel, 3 total delete,
                4 execution of a VISIBLE limit order, 5 execution of a
                HIDDEN order, 6 cross/auction trade, 7 trading halt
      Size      shares
      Price     integer, units of 1e-4 USD  (5859800 -> $585.98)
      Direction for types 1/2/3: +1 buy limit order, -1 sell limit order

orderbook file  _orderbook_<level>.csv
    AskPrice1, AskSize1, BidPrice1, BidSize1, AskPrice2, ... (4 cols/level)
    Prices integer in 1e-4 USD; a missing level is padded with +/-9999999999.

TRADE SIGN — the classic gotcha this module handles for you
-----------------------------------------------------------
For an execution (Type 4), `Direction` describes the side of the resting
LIMIT order that was consumed, NOT the side of the aggressor. A Type-4 row
with Direction = +1 means a resting BUY limit order was executed, i.e. the
aggressor was a SELLER. Signed order flow (needed for permanent-impact
estimation) therefore uses  signed_size = -Direction * Size.

Getting this backwards flips the sign of the estimated permanent impact
coefficient, which is why `calibrate.py` asserts gamma > 0 and
`test_calibration.py` includes a deliberate sign-flip regression test.

Everything returned by this module is in natural units: prices in dollars,
time in seconds after midnight, sizes in shares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRICE_SCALE = 1e-4          # LOBSTER integer price -> dollars
PAD_THRESHOLD = 1e8         # |price| above this marks an unoccupied level

MSG_COLS = ["time", "type", "order_id", "size", "price", "direction"]

TYPE_NEW, TYPE_CANCEL_PART, TYPE_DELETE = 1, 2, 3
TYPE_EXEC_VISIBLE, TYPE_EXEC_HIDDEN, TYPE_CROSS, TYPE_HALT = 4, 5, 6, 7


@dataclass
class LOBData:
    """Normalised LOBSTER data for one ticker-day."""

    messages: pd.DataFrame          # time, type, size, price ($), direction
    book: pd.DataFrame              # ask_px_1..N, ask_sz_1..N, bid_px_1..N, bid_sz_1..N
    n_levels: int
    ticker: str = ""
    date: str = ""

    # ---------------------------------------------------------- derived
    @property
    def best_ask(self) -> np.ndarray:
        return self.book["ask_px_1"].to_numpy()

    @property
    def best_bid(self) -> np.ndarray:
        return self.book["bid_px_1"].to_numpy()

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.best_ask + self.best_bid)

    @property
    def spread(self) -> np.ndarray:
        return self.best_ask - self.best_bid

    @property
    def time(self) -> np.ndarray:
        return self.messages["time"].to_numpy()

    def bid_ladder(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(prices, sizes) of the bid side at snapshot i, best first,
        unoccupied levels dropped."""
        px = np.array([self.book[f"bid_px_{l}"].iloc[i] for l in range(1, self.n_levels + 1)])
        sz = np.array([self.book[f"bid_sz_{l}"].iloc[i] for l in range(1, self.n_levels + 1)])
        ok = (np.abs(px) < PAD_THRESHOLD) & (sz > 0)
        return px[ok], sz[ok]

    def ask_ladder(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        px = np.array([self.book[f"ask_px_{l}"].iloc[i] for l in range(1, self.n_levels + 1)])
        sz = np.array([self.book[f"ask_sz_{l}"].iloc[i] for l in range(1, self.n_levels + 1)])
        ok = (np.abs(px) < PAD_THRESHOLD) & (sz > 0)
        return px[ok], sz[ok]

    def signed_flow(self) -> pd.DataFrame:
        """Executions with aggressor-signed size (+ = buyer-initiated).

        See the TRADE SIGN note in this module's docstring: for Type 4 the
        message `direction` is the side of the RESTING order, so the
        aggressor sign is its negation.
        """
        m = self.messages
        ex = m[m["type"].isin([TYPE_EXEC_VISIBLE, TYPE_EXEC_HIDDEN])].copy()
        ex["signed_size"] = -ex["direction"] * ex["size"]
        return ex[["time", "price", "size", "signed_size"]]


def _book_columns(n_levels: int) -> list[str]:
    cols = []
    for l in range(1, n_levels + 1):
        cols += [f"ask_px_{l}", f"ask_sz_{l}", f"bid_px_{l}", f"bid_sz_{l}"]
    return cols


def load_lobster(message_path: str, orderbook_path: str, n_levels: int,
                 ticker: str = "", date: str = "",
                 trading_hours_only: bool = True) -> LOBData:
    """Load a LOBSTER ticker-day into natural units.

    trading_hours_only clips to 09:30:00-16:00:00 (34200-57600 s), dropping
    the opening/closing auction artefacts that otherwise dominate impact fits.
    """
    msg = pd.read_csv(message_path, header=None, names=MSG_COLS)
    book = pd.read_csv(orderbook_path, header=None, names=_book_columns(n_levels))
    if len(msg) != len(book):
        raise ValueError(
            f"message/orderbook row mismatch: {len(msg)} vs {len(book)} — "
            "the two files must be the same ticker, day and level count"
        )

    msg = msg.copy()
    msg["price"] = msg["price"] * PRICE_SCALE
    for l in range(1, n_levels + 1):
        book[f"ask_px_{l}"] = book[f"ask_px_{l}"] * PRICE_SCALE
        book[f"bid_px_{l}"] = book[f"bid_px_{l}"] * PRICE_SCALE

    if trading_hours_only:
        keep = (msg["time"] >= 34200.0) & (msg["time"] <= 57600.0)
        msg, book = msg[keep].reset_index(drop=True), book[keep].reset_index(drop=True)

    # drop snapshots with a crossed or empty top of book (halts, pad rows)
    ok = (book["ask_px_1"] > book["bid_px_1"]) & \
         (np.abs(book["ask_px_1"]) < PAD_THRESHOLD) & \
         (np.abs(book["bid_px_1"]) < PAD_THRESHOLD)
    msg, book = msg[ok].reset_index(drop=True), book[ok].reset_index(drop=True)

    return LOBData(messages=msg, book=book, n_levels=n_levels,
                   ticker=ticker, date=date)


def resample_grid(lob: LOBData, interval_s: float) -> pd.DataFrame:
    """Snap the event-time book onto a regular time grid.

    Returns one row per interval with the LAST snapshot index in that
    interval, its mid and spread, and the net signed flow over the interval.
    Event-time data is irregular; impact and volatility estimators need a
    clock-time grid.
    """
    t = lob.time
    t0, t1 = t[0], t[-1]
    edges = np.arange(t0, t1 + interval_s, interval_s)
    idx = np.searchsorted(t, edges[1:], side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)

    flow = lob.signed_flow()
    net = np.zeros(len(edges) - 1)
    if len(flow):
        bins = np.clip(np.searchsorted(edges, flow["time"].to_numpy(), side="right") - 1,
                       0, len(edges) - 2)
        np.add.at(net, bins, flow["signed_size"].to_numpy())

    mid = lob.mid
    return pd.DataFrame({
        "t_end": edges[1:],
        "snap_idx": idx,
        "mid": mid[idx],
        "spread": lob.spread[idx],
        "net_signed_size": net,
    })
