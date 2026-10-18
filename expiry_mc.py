"""Expiry-day Monte Carlo simulator for Nifty & Sensex option positions.

Model (per path, per index)
  * correlated Gaussian shocks across indices (Cholesky of the correlation matrix)
  * stochastic path volatility  m = exp(v*N - v^2)   (E[m^2] = 1)  -> fat-tailed settlement distribution
  * optional market-wide jump (shared by all indices) with compensated drift
  * log-returns on an intraday grid; time runs in *trading* years (minutes/375/252) so that
    sigma^2 * T equals the variance actually remaining in the session

Evaluation
  * each leg is repriced with Black-Scholes along every path (intrinsic at the final bar =
    cash settlement), so leg-level stop-losses and a portfolio-level rupee stop are simulated
    with path dependence rather than assumed away
  * output: P&L distribution, win rate, VaR/CVaR, worst case, breach probabilities

Simplifications (be honest about them in your write-up): constant implied vol when repricing
(no intraday smile dynamics), stop fills at the trigger level plus a slippage fraction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import INDEX_SPECS, RISK_FREE_RATE, TRADING_DAYS, SESSION_MINUTES
from ..pricing import bs_price, MIN_T


@dataclass
class IndexSpec:
    name: str
    spot: float
    iv: float                       # annualised implied vol used for pricing/repricing
    lot_size: int


@dataclass
class Leg:
    index: str
    opt: str                        # "C" / "P"
    strike: float
    lots: int                       # negative = short
    entry_price: Optional[float] = None   # None -> BS price at entry


def simulate_paths(specs: dict[str, IndexSpec], corr: float = 0.97, n_paths: int = 20000, n_steps: int = 75,
                   minutes_remaining: float = 370, rv_to_iv: float = 0.85, volvol: float = 0.35,
                   jump_prob: float = 0.03, jump_sd: float = 0.012, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = list(specs)
    k = len(names)
    C = np.full((k, k), corr)
    np.fill_diagonal(C, 1.0)
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((n_paths, n_steps, k)) @ L.T

    day_frac = minutes_remaining / SESSION_MINUTES
    dt_frac = day_frac / n_steps
    m = np.exp(volvol * rng.standard_normal(n_paths) - volvol**2)[:, None]       # path vol multiplier

    has_jump = rng.random(n_paths) < jump_prob
    jump_step = rng.integers(0, n_steps, n_paths)
    jump_size = rng.normal(0.0, jump_sd, n_paths) * has_jump
    comp = jump_prob * (np.exp(0.5 * jump_sd**2) - 1.0) / n_steps               # keep E[S_T] = S_0

    paths = {}
    for j, nme in enumerate(names):
        sp = specs[nme]
        sig_step = sp.iv * rv_to_iv * np.sqrt(dt_frac / TRADING_DAYS)
        ret = -0.5 * (sig_step * m) ** 2 + sig_step * m * z[:, :, j] - comp
        ret[np.arange(n_paths), jump_step] += jump_size
        logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(ret, axis=1)], axis=1)
        paths[nme] = sp.spot * np.exp(logS)
    return paths


def evaluate(legs: list[Leg], specs: dict[str, IndexSpec], paths: dict[str, np.ndarray],
             minutes_remaining: float = 370, r: float = RISK_FREE_RATE, leg_sl_mult: Optional[float] = None,
             portfolio_sl_inr: Optional[float] = None, slippage: float = 0.02, cost_per_lot: float = 40.0) -> dict:
    n_paths, n1 = next(iter(paths.values())).shape
    n_steps = n1 - 1
    T_grid = (minutes_remaining / SESSION_MINUTES / TRADING_DAYS) * (1 - np.arange(n1) / n_steps)

    total = np.zeros((n_paths, n1))
    idx = np.arange(n1)[None, :]
    costs = 0.0
    for lg in legs:
        sp, S = specs[lg.index], paths[lg.index]
        px = np.empty((n_paths, n1))
        px[:, :-1] = bs_price(S[:, :-1], lg.strike, np.maximum(T_grid[:-1], MIN_T)[None, :], r, sp.iv, lg.opt)
        px[:, -1] = np.maximum(S[:, -1] - lg.strike, 0) if lg.opt == "C" else np.maximum(lg.strike - S[:, -1], 0)
        entry = lg.entry_price if lg.entry_price is not None else float(px[0, 0])
        units = lg.lots * sp.lot_size
        mtm = (px - entry) * units
        if leg_sl_mult and lg.lots < 0:
            trig = px >= entry * leg_sl_mult
            has, first = trig.any(1), trig.argmax(1)
            locked = (entry * leg_sl_mult * (1 + slippage) - entry) * units          # negative for a short
            mtm = np.where(has[:, None] & (idx >= first[:, None]), locked, mtm)
        total += mtm
        costs += cost_per_lot * abs(lg.lots)

    final = total[:, -1].copy()
    if portfolio_sl_inr:
        trig = total <= -portfolio_sl_inr
        has = trig.any(1)
        final = np.where(has, -portfolio_sl_inr * (1 + slippage), final)
    pnl = final - costs
    return dict(pnl=pnl, stats=_stats(pnl), breach=_breach(legs, paths))


def _stats(p: np.ndarray, alpha: float = 0.05) -> dict:
    var = np.quantile(p, alpha)
    return dict(mean=float(p.mean()), median=float(np.median(p)), std=float(p.std()),
                win_rate=float((p > 0).mean()), var_95=float(var), cvar_95=float(p[p <= var].mean()),
                worst=float(p.min()), best=float(p.max()),
                p_loss_gt_1x_mean_win=float((p < -abs(p[p > 0].mean() if (p > 0).any() else 0)).mean()))


def _breach(legs, paths) -> dict:
    out = {}
    for lg in legs:
        if lg.lots >= 0:
            continue
        S_T = paths[lg.index][:, -1]
        out[f"{lg.index} {int(lg.strike)}{lg.opt}E ITM at expiry"] = float((S_T > lg.strike).mean() if lg.opt == "C" else (S_T < lg.strike).mean())
    return out


def run(config: dict) -> dict:
    specs = {s["name"]: IndexSpec(**s) for s in config["indices"]}
    legs = [Leg(**l) for l in config["legs"]]
    sim = config.get("sim", {})
    mins = sim.get("minutes_remaining", 370)
    paths = simulate_paths(specs, minutes_remaining=mins, **{k: v for k, v in sim.items() if k != "minutes_remaining"})
    return evaluate(legs, specs, paths, minutes_remaining=mins, **config.get("risk", {})) | dict(paths=paths)


DEMO = {
    "indices": [dict(name="NIFTY", spot=24500, iv=0.14, lot_size=INDEX_SPECS["NIFTY"]["lot_size"]),
                dict(name="SENSEX", spot=80500, iv=0.14, lot_size=INDEX_SPECS["SENSEX"]["lot_size"])],
    "legs": [dict(index="NIFTY", opt="C", strike=24700, lots=-4), dict(index="NIFTY", opt="P", strike=24300, lots=-4),
             dict(index="NIFTY", opt="C", strike=24900, lots=4), dict(index="NIFTY", opt="P", strike=24100, lots=4),
             dict(index="SENSEX", opt="C", strike=81200, lots=-6), dict(index="SENSEX", opt="P", strike=79800, lots=-6)],
    "sim": {"minutes_remaining": 370, "n_paths": 20000, "corr": 0.97},
    "risk": {"leg_sl_mult": 2.0, "portfolio_sl_inr": 25000, "slippage": 0.02, "cost_per_lot": 40},
}


def main():
    ap = argparse.ArgumentParser(description="Expiry-day Monte Carlo.")
    ap.add_argument("--config", help="JSON file (see examples/expiry_config.json); omit for demo")
    ap.add_argument("--plot", help="save P&L histogram PNG to this path")
    a = ap.parse_args()
    cfg = json.load(open(a.config)) if a.config else DEMO
    res = run(cfg)
    print("P&L distribution (INR):")
    for k, v in res["stats"].items():
        print(f"  {k:>24}: {v:,.3f}" if k in ("win_rate", "p_loss_gt_1x_mean_win") else f"  {k:>24}: {v:,.0f}")
    print("Short-strike breach probability:")
    for k, v in res["breach"].items():
        print(f"  {k:>34}: {v:.1%}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        plt.hist(res["pnl"], bins=80, color="#3b6ea5")
        plt.axvline(res["stats"]["var_95"], color="crimson", ls="--", label="VaR 95%")
        plt.axvline(res["stats"]["mean"], color="black", label="mean")
        plt.xlabel("Expiry-day P&L (INR)"); plt.ylabel("paths"); plt.legend(); plt.tight_layout()
        plt.savefig(a.plot, dpi=140)
        print("saved", a.plot)


if __name__ == "__main__":
    main()
