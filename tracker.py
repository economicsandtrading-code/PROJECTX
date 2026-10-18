"""Live mark-to-market P&L tracker for index options using Zerodha KiteConnect.

Flow
----
1. Poll ``kite.trades()`` (today's fills) -> rebuild the FIFO book (idempotent, so re-polling is safe).
2. Subscribe to the traded instruments' tokens on the KiteTicker WebSocket (LTP mode).
3. Every ``--refresh`` seconds print realised / unrealised / net P&L per symbol and in total.

Auth: set KITE_API_KEY and KITE_ACCESS_TOKEN (access tokens expire daily -- see README).

Offline replay (no API needed):
    python -m quantlab.mtm.tracker --trades-csv examples/sample_trades.csv --ltp-csv examples/sample_ltp.csv
"""
from __future__ import annotations

import argparse
import os
import re
import threading
import time

import pandas as pd

from .fifo import Fill, FifoBook, book_from_fills, fills_from_csv


def underlying_of(tradingsymbol: str) -> str:
    m = re.match(r"[A-Za-z]+", tradingsymbol)
    return m.group(0).upper() if m else ""


def fills_from_kite_trades(trades: list, underlyings: set[str]) -> list[Fill]:
    fills = []
    for t in trades:
        sym = t["tradingsymbol"]
        if underlyings and underlying_of(sym) not in underlyings:
            continue
        sign = 1 if t["transaction_type"].upper() == "BUY" else -1
        ts = t.get("fill_timestamp") or t.get("exchange_timestamp") or t.get("order_timestamp")
        fills.append(Fill(ts=ts, symbol=sym, qty=sign * int(t["quantity"]),
                          price=float(t["average_price"]), token=t.get("instrument_token")))
    return fills


class LiveMTM:
    def __init__(self, api_key: str, access_token: str, underlyings: set[str],
                 refresh: float = 1.0, poll_trades_every: float = 10.0):
        from kiteconnect import KiteConnect, KiteTicker   # imported lazily so tests/offline mode work

        self.underlyings = underlyings
        self.refresh, self.poll_every = refresh, poll_trades_every
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.ws = KiteTicker(api_key, access_token)
        self.ltp: dict[str, float] = {}
        self.tok2sym: dict[int, str] = {}
        self.book = FifoBook()
        self._lock = threading.Lock()
        self._subscribed: set[int] = set()
        self._connected = False

        self.ws.on_connect = self._on_connect
        self.ws.on_ticks = self._on_ticks
        self.ws.on_close = lambda ws, code, reason: print(f"[ws] closed {code} {reason}")
        self.ws.on_error = lambda ws, code, reason: print(f"[ws] error {code} {reason}")
        self.ws.on_reconnect = lambda ws, n: print(f"[ws] reconnecting (attempt {n})")

    # ---- websocket callbacks (run on ticker thread) -------------------------------------------
    def _on_connect(self, ws, response):
        self._connected = True
        self._subscribe(list(self.tok2sym))

    def _on_ticks(self, ws, ticks):
        with self._lock:
            for t in ticks:
                sym = self.tok2sym.get(t["instrument_token"])
                if sym is not None and t.get("last_price") is not None:
                    self.ltp[sym] = t["last_price"]

    def _subscribe(self, tokens):
        new = [t for t in tokens if t not in self._subscribed]
        if new and self._connected:
            self.ws.subscribe(new)
            self.ws.set_mode(self.ws.MODE_LTP, new)
            self._subscribed.update(new)

    # ---- trades -------------------------------------------------------------------------------
    def refresh_trades(self):
        fills = fills_from_kite_trades(self.kite.trades(), self.underlyings)
        book = book_from_fills(fills)
        with self._lock:
            self.book = book
            for sym, tok in book.tokens.items():
                self.tok2sym[tok] = sym
        self._subscribe(list(self.tok2sym))

    def seed_ltp_via_rest(self):
        """Fallback so rows show a price before the first tick arrives."""
        syms = [f"NFO:{s}" if not s.startswith("SENSEX") else f"BFO:{s}" for s in self.book.symbols()]
        if not syms:
            return
        for k, v in self.kite.ltp(syms).items():
            self.ltp.setdefault(k.split(":", 1)[1], v["last_price"])

    # ---- main loop ----------------------------------------------------------------------------
    def run(self):
        self.ws.connect(threaded=True)
        last_poll = 0.0
        try:
            while True:
                if time.time() - last_poll > self.poll_every:
                    self.refresh_trades()
                    self.seed_ltp_via_rest()
                    last_poll = time.time()
                with self._lock:
                    snap = self.book.snapshot(dict(self.ltp))
                render(snap)
                time.sleep(self.refresh)
        except KeyboardInterrupt:
            print("\nstopped.")
            self.ws.close()


def render(snap: pd.DataFrame, clear: bool = True):
    if clear:
        print("\033[H\033[J", end="")
    print(time.strftime("%H:%M:%S"), "| live MTM")
    if snap.empty:
        print("no trades yet")
        return
    show = snap.copy()
    for c in ("realized", "unrealized", "fees", "net_pnl"):
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    print(show.to_string(index=False))
    tot = snap[["realized", "unrealized", "fees", "net_pnl"]].sum()
    print("-" * 78)
    print(f"TOTAL  realized {tot.realized:,.0f} | unrealized {tot.unrealized:,.0f} | "
          f"fees {tot.fees:,.0f} | NET {tot.net_pnl:,.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--underlyings", default="NIFTY,SENSEX")
    ap.add_argument("--refresh", type=float, default=1.0)
    ap.add_argument("--trades-csv", help="offline replay: fills CSV")
    ap.add_argument("--ltp-csv", help="offline replay: symbol,ltp CSV")
    a = ap.parse_args()
    unds = {u.strip().upper() for u in a.underlyings.split(",") if u.strip()}

    if a.trades_csv:
        book = book_from_fills([f for f in fills_from_csv(a.trades_csv)
                                if not unds or underlying_of(f.symbol) in unds])
        ltp = dict(pd.read_csv(a.ltp_csv).values) if a.ltp_csv else {}
        render(book.snapshot(ltp), clear=False)
        return

    key, tok = os.getenv("KITE_API_KEY"), os.getenv("KITE_ACCESS_TOKEN")
    if not key or not tok:
        raise SystemExit("Set KITE_API_KEY and KITE_ACCESS_TOKEN (see .env.example).")
    LiveMTM(key, tok, unds, refresh=a.refresh).run()


if __name__ == "__main__":
    main()
