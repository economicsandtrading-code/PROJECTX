"""Cross-index options portfolio optimiser (Black-Scholes): maximise net theta s.t. Greeks limits.

Key observation: net Greeks, net theta and scenario P&L are all LINEAR in position sizes, so the
problem is a mixed-integer linear program (integer lots) -- solved globally with scipy.optimize.milp
(HiGHS), no heuristics or local-minimum issues.

    maximise    sum_i  x_i * theta_i  -  cost * sum_i |x_i|            (theta in INR/day per lot)
    subject to  |sum x_i * delta_i|          <= max_abs_delta_inr      (rupee delta, notional)
                sum x_i * gamma1pct_i        >= -max_short_gamma_inr    (P&L of a 1% move from gamma)
                |sum x_i * vega_i|           <= max_abs_vega_inr        (INR per vol point)
                sum x_i * pnl_i(s)           >= -max_scenario_loss_inr  for every spot/vol shock s
                sum |x_i|                    <= max_total_lots
                x_i integer, -max_short <= x_i <= max_long

x_i = long_i - short_i (both >= 0) so |x_i| is linear. Long wings appear automatically whenever
the scenario-loss constraint makes buying protection cheaper than shrinking the short book.
Scenarios apply the *same* % spot shock to both indices (beta = 1) with a parallel vol shock.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from ..config import INDEX_SPECS, RISK_FREE_RATE, DIVIDEND_YIELD
from ..pricing import bs_greeks, bs_price


@dataclass
class Underlying:
    name: str
    spot: float
    dtes: list[int]                      # expiries (days) to consider
    atm_iv: float | list[float]          # one IV per expiry, or a scalar
    lot_size: int = 0
    strike_step: int = 0

    def __post_init__(self):
        spec = INDEX_SPECS.get(self.name, {})
        self.lot_size = self.lot_size or spec.get("lot_size", 1)
        self.strike_step = self.strike_step or spec.get("strike_step", 50)
        if np.isscalar(self.atm_iv):
            self.atm_iv = [float(self.atm_iv)] * len(self.dtes)


@dataclass
class Limits:
    max_abs_delta_inr: float = 1_500_000
    max_short_gamma_inr: float = 15_000
    max_abs_vega_inr: float = 20_000
    max_scenario_loss_inr: float = 60_000
    max_total_lots: int = 80
    max_short_per_leg: int = 20
    max_long_per_leg: int = 20
    cost_per_lot: float = 20.0
    scenarios: list = field(default_factory=lambda: [      # (spot shock, vol shock in abs vol pts / 100)
        (-0.05, +0.06), (-0.03, +0.03), (-0.015, +0.01), (0.015, -0.01), (0.03, -0.01), (0.05, +0.02)])


def build_candidates(unds: list[Underlying], width: float = 0.08, skew: float = 1.5, min_premium: float = 1.0,
                     r: float = RISK_FREE_RATE, q: float = DIVIDEND_YIELD, limits: Limits = Limits()) -> pd.DataFrame:
    """OTM calls and puts on a strike grid with a linear moneyness skew: iv = atm*(1 + skew*ln(S/K))."""
    rows = []
    for u in unds:
        S, step = u.spot, u.strike_step
        lo, hi = int(np.floor(S * (1 - width) / step)) * step, int(np.ceil(S * (1 + width) / step)) * step
        for dte, atm in zip(u.dtes, u.atm_iv):
            T = dte / 365.0
            for K in range(lo, hi + step, step):
                for opt in ("C", "P"):
                    if (opt == "C" and K < S) or (opt == "P" and K > S):
                        continue                                    # OTM only
                    iv = atm * float(np.clip(1 + skew * np.log(S / K), 0.6, 2.0))
                    px = bs_price(S, K, T, r, iv, opt, q)
                    if px < min_premium:
                        continue
                    g = bs_greeks(S, K, T, r, iv, opt, q)
                    lot = u.lot_size
                    pnl_s = [(bs_price(S * (1 + ds), K, T, r, max(iv + dv, 0.02), opt, q) - px) * lot
                             for ds, dv in limits.scenarios]
                    rows.append(dict(underlying=u.name, dte=dte, opt=opt, strike=K, iv=iv, price=px,
                                     delta_inr=g["delta"] * S * lot,
                                     gamma_inr=0.5 * g["gamma"] * (0.01 * S) ** 2 * lot,
                                     vega_inr=g["vega"] * lot, theta_inr=g["theta"] * lot,
                                     lot_size=lot, **{f"s{j}": v for j, v in enumerate(pnl_s)}))
    return pd.DataFrame(rows)


def optimise(cands: pd.DataFrame, lim: Limits = Limits(), time_limit: float = 30.0) -> tuple[pd.DataFrame, dict]:
    n = len(cands)
    sc_cols = [c for c in cands.columns if c.startswith("s") and c[1:].isdigit()]
    # variable vector: [long_0..long_n-1, short_0..short_n-1];  x = long - short
    def row(vec):
        return np.concatenate([vec, -vec])

    theta = cands["theta_inr"].to_numpy()
    c = np.concatenate([-theta + lim.cost_per_lot, theta + lim.cost_per_lot])      # milp minimises

    A, lb, ub = [], [], []
    d = cands["delta_inr"].to_numpy()
    A.append(row(d)); lb.append(-lim.max_abs_delta_inr); ub.append(lim.max_abs_delta_inr)
    A.append(row(cands["gamma_inr"].to_numpy())); lb.append(-lim.max_short_gamma_inr); ub.append(np.inf)
    v = cands["vega_inr"].to_numpy()
    A.append(row(v)); lb.append(-lim.max_abs_vega_inr); ub.append(lim.max_abs_vega_inr)
    for col in sc_cols:
        A.append(row(cands[col].to_numpy())); lb.append(-lim.max_scenario_loss_inr); ub.append(np.inf)
    A.append(np.ones(2 * n)); lb.append(0); ub.append(lim.max_total_lots)

    res = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.ones(2 * n),
               bounds=Bounds(np.zeros(2 * n), np.r_[np.full(n, lim.max_long_per_leg), np.full(n, lim.max_short_per_leg)]),
               options={"time_limit": time_limit, "mip_rel_gap": 0.01})
    if res.x is None:
        raise RuntimeError(f"optimiser failed: {res.message}")
    x = np.round(res.x[:n] - res.x[n:]).astype(int)

    pos = cands.assign(lots=x)
    pos = pos[pos["lots"] != 0].copy()
    pos["units"] = pos["lots"] * pos["lot_size"]
    out = pos[["underlying", "dte", "opt", "strike", "iv", "price", "lots", "units"]].sort_values(["underlying", "dte", "opt", "strike"])
    xs = pos["lots"].to_numpy()
    summary = dict(
        status=res.message,
        net_theta_inr_day=float((pos["theta_inr"] * xs).sum()),
        net_delta_inr=float((pos["delta_inr"] * xs).sum()),
        net_gamma_1pct_inr=float((pos["gamma_inr"] * xs).sum()),
        net_vega_inr=float((pos["vega_inr"] * xs).sum()),
        net_premium_inr=float(-(pos["price"] * pos["units"]).sum()),       # + = credit
        worst_scenario_pnl_inr=float(min((pos[c] * xs).sum() for c in sc_cols)) if len(pos) else 0.0,
        total_lots=int(np.abs(xs).sum()),
    )
    return out.reset_index(drop=True), summary


def main():
    ap = argparse.ArgumentParser(description="Cross-index theta-maximising portfolio (MILP).")
    ap.add_argument("--nifty", type=float, default=24500)
    ap.add_argument("--sensex", type=float, default=80500)
    ap.add_argument("--dte", default="2,9", help="expiries (days) considered for both indices")
    ap.add_argument("--iv", type=float, default=0.14)
    ap.add_argument("--max-loss", type=float, default=60000, help="max loss across stress scenarios (INR)")
    ap.add_argument("--max-vega", type=float, default=20000)
    ap.add_argument("--max-delta", type=float, default=1_500_000)
    ap.add_argument("--time-limit", type=float, default=30.0)
    a = ap.parse_args()
    dtes = [int(x) for x in a.dte.split(",")]
    unds = [Underlying("NIFTY", a.nifty, dtes, a.iv), Underlying("SENSEX", a.sensex, dtes, a.iv)]
    lim = Limits(max_scenario_loss_inr=a.max_loss, max_abs_vega_inr=a.max_vega, max_abs_delta_inr=a.max_delta)
    cands = build_candidates(unds, limits=lim)
    pos, summ = optimise(cands, lim, a.time_limit)
    print(f"{len(cands)} candidate contracts\n")
    print(pos.round(3).to_string(index=False))
    print()
    for k, v in summ.items():
        print(f"{k:>26}: {v:,.1f}" if isinstance(v, float) else f"{k:>26}: {v}")


if __name__ == "__main__":
    main()
