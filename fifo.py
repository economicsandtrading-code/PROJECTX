"""FIFO P&L reconstruction for a stream of fills.

Quantities are signed *units* (buy > 0, sell < 0). Kite reports quantity in units already
(lots x lot_size), so no extra multiplier is required.

Realised P&L when a fill closes an existing lot of signed size s * m:
    pnl = sign(lot) * (fill_price - lot_price) * m
which is correct for both closing longs (sell above cost) and closing shorts (buy below cost).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Fill:
    ts: object
    symbol: str
    qty: int                       # signed: +buy / -sell
    price: float
    fees: float = 0.0
    token: Optional[int] = None


@dataclass
class FifoBook:
    _lots: dict = field(default_factory=lambda: defaultdict(deque))
    realized: dict = field(default_factory=lambda: defaultdict(float))
    fees: dict = field(default_factory=lambda: defaultdict(float))
    tokens: dict = field(default_factory=dict)
    closed: list = field(default_factory=list)    # audit trail of matched lots

    def add_fill(self, f: Fill) -> None:
        if f.qty == 0:
            return
        lots = self._lots[f.symbol]
        self.fees[f.symbol] += f.fees
        if f.token is not None:
            self.tokens[f.symbol] = f.token
        remaining = int(f.qty)
        while remaining != 0 and lots and (lots[0][0] > 0) != (remaining > 0):
            lot = lots[0]
            m = min(abs(remaining), abs(lot[0]))
            s = 1 if lot[0] > 0 else -1
            pnl = s * (f.price - lot[1]) * m
            self.realized[f.symbol] += pnl
            self.closed.append(dict(ts=f.ts, symbol=f.symbol, qty=m, side="long" if s > 0 else "short",
                                    open_price=lot[1], close_price=f.price, pnl=pnl))
            lot[0] -= s * m
            remaining += s * m
            if lot[0] == 0:
                lots.popleft()
        if remaining != 0:
            lots.append([remaining, f.price])

    # ---- queries -------------------------------------------------------------------------
    def symbols(self):
        return sorted(set(self._lots) | set(self.realized))

    def net_qty(self, symbol: str) -> int:
        return sum(l[0] for l in self._lots.get(symbol, ()))

    def avg_price(self, symbol: str) -> float:
        lots = self._lots.get(symbol, ())
        q = sum(abs(l[0]) for l in lots)
        return sum(abs(l[0]) * l[1] for l in lots) / q if q else 0.0

    def unrealized(self, symbol: str, ltp: float) -> float:
        return sum(l[0] * (ltp - l[1]) for l in self._lots.get(symbol, ()))

    def snapshot(self, ltp: dict) -> pd.DataFrame:
        rows = []
        for s in self.symbols():
            q = self.net_qty(s)
            px = ltp.get(s)
            unreal = self.unrealized(s, px) if (q and px is not None) else 0.0
            rows.append(dict(symbol=s, net_qty=q, avg_price=round(self.avg_price(s), 2),
                             ltp=px, realized=self.realized.get(s, 0.0), unrealized=unreal,
                             fees=self.fees.get(s, 0.0)))
        df = pd.DataFrame(rows, columns=["symbol", "net_qty", "avg_price", "ltp", "realized", "unrealized", "fees"])
        df["net_pnl"] = df["realized"] + df["unrealized"] - df["fees"]
        return df


def book_from_fills(fills) -> FifoBook:
    book = FifoBook()
    for f in sorted(fills, key=lambda x: str(x.ts)):
        book.add_fill(f)
    return book


def fills_from_csv(path: str) -> list[Fill]:
    """CSV columns: timestamp,symbol,side,qty,price[,fees]  (side = BUY/SELL, qty unsigned)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    out = []
    for r in df.itertuples():
        sign = 1 if str(r.side).upper().startswith("B") else -1
        out.append(Fill(ts=r.timestamp, symbol=r.symbol, qty=sign * int(r.qty), price=float(r.price),
                        fees=float(getattr(r, "fees", 0.0) or 0.0)))
    return out
