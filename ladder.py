"""Cross-expiry laddering for options selling.

Given spot, a regime policy and a set of expiries, build a ladder of short strikes:
  * the target delta shrinks for later expiries (more vega/gamma-time per rupee of premium)
  * lots are weighted towards near expiries (faster theta) but spread so no single expiry
    carries the whole book -> smoother P&L and roll-down when the front expiry settles
  * in a TRENDING regime the threatened side (calls in an up-trend, puts in a down-trend)
    is pushed further OTM
Flat IV per expiry is used to invert delta -> strike (closed form); premiums/greeks are then
recomputed at the rounded strike.
"""
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import INDEX_SPECS, RISK_FREE_RATE, DIVIDEND_YIELD
from ..pricing import bs_greeks, bs_price, strike_for_delta
from .classifier import Regime, POLICY


def build_ladder(spot: float, regime: Regime, dtes: Sequence[int], atm_iv: float | Sequence[float],
                 base_lots: int = 10, underlying: str = "NIFTY", delta_decay: float = 0.85,
                 r: float = RISK_FREE_RATE, q: float = DIVIDEND_YIELD) -> pd.DataFrame:
    spec = INDEX_SPECS[underlying]
    step, lot_size = spec["strike_step"], spec["lot_size"]
    ivs = [atm_iv] * len(dtes) if np.isscalar(atm_iv) else list(atm_iv)
    pol = regime.policy
    weights = np.array([1.0 / (1 + i) for i in range(len(dtes))])
    weights /= weights.sum()
    total_lots = base_lots * pol["size_mult"]

    rows = []
    for i, (dte, iv) in enumerate(zip(dtes, ivs)):
        T = dte / 365.0
        d_target = pol["short_delta"] * delta_decay**i
        lots_leg = max(1, int(round(total_lots * weights[i] / 2)))
        for opt in ("C", "P"):
            d = d_target
            if regime.label == "TRENDING":
                threatened = "C" if regime.trend_side > 0 else "P"
                if opt == threatened:
                    d *= 0.6
            K = strike_for_delta(spot, T, r, iv, d, opt, q)
            K = round(K / step) * step
            g = bs_greeks(spot, K, T, r, iv, opt, q)
            rows.append(dict(dte=dte, leg=f"{int(K)}{'CE' if opt == 'C' else 'PE'}", opt=opt, strike=K,
                             iv=iv, delta=round(g["delta"], 3), premium=round(bs_price(spot, K, T, r, iv, opt, q), 2),
                             theta_day=round(g["theta"], 2), vega=round(g["vega"], 2), lots=-lots_leg,
                             lot_size=lot_size))
    df = pd.DataFrame(rows)
    df["theta_inr_day"] = df["lots"] * df["lot_size"] * df["theta_day"]   # short lots x negative theta => +
    df["premium_inr"] = -df["lots"] * df["lot_size"] * df["premium"]
    return df


def summarize(df: pd.DataFrame) -> dict:
    u = -df["lots"] * df["lot_size"]
    return dict(premium_collected=float((u * df["premium"]).sum()),
                theta_per_day=float(-(u * df["theta_day"]).sum()),
                net_delta_units=float((-u * df["delta"]).sum()),
                net_vega_per_pt=float((-u * df["vega"]).sum()))


def main():
    ap = argparse.ArgumentParser(description="Cross-expiry short-option ladder.")
    ap.add_argument("--spot", type=float, default=24500)
    ap.add_argument("--iv", type=float, default=0.14)
    ap.add_argument("--dte", default="2,9,16", help="comma-separated days to expiry")
    ap.add_argument("--regime", default="RANGE_NORMAL", choices=list(POLICY))
    ap.add_argument("--trend-side", type=int, default=0)
    ap.add_argument("--lots", type=int, default=10)
    ap.add_argument("--underlying", default="NIFTY", choices=list(INDEX_SPECS))
    a = ap.parse_args()
    reg = Regime(a.regime, {}, dict(POLICY[a.regime]), a.trend_side)
    df = build_ladder(a.spot, reg, [int(x) for x in a.dte.split(",")], a.iv, a.lots, a.underlying)
    print(df.drop(columns=["opt"]).to_string(index=False))
    print("\n", {k: round(v, 1) for k, v in summarize(df).items()})


if __name__ == "__main__":
    main()
