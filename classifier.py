"""Rule-based market regime classifier for options-selling decisions.

Features (daily bars)
  rv5 / rv20      close-to-close realised vol (annualised)
  park10          Parkinson (high-low) vol, 10d
  vol_ratio       rv5 / rv20  -> vol expansion / compression
  eff_ratio       Kaufman efficiency ratio (20d): |net move| / path length  -> trendiness
  trend_z         (close - EMA20) / ATR14                                  -> stretch from trend
  vol_pct         1y percentile rank of VIX (or rv20 if no VIX supplied)
  vrp             VIX/100 - rv20 (variance-risk-premium proxy; the 'edge' a seller is paid)

Regimes -> each maps to a policy (target short delta, size multiplier, structure guidance).
Thresholds live in RegimeConfig so they can be tuned / walk-forward tested.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


@dataclass
class RegimeConfig:
    stress_vol_pct: float = 0.85
    stress_vol_ratio: float = 1.60
    trend_er: float = 0.35
    trend_z: float = 1.00
    calm_vol_pct: float = 0.35
    calm_er: float = 0.25


POLICY = {
    "HIGH_VOL_STRESS": dict(short_delta=0.08, size_mult=0.40, structure="defined-risk only (iron condor / hedged strangle)"),
    "TRENDING":        dict(short_delta=0.12, size_mult=0.60, structure="skew strikes: push the threatened side further OTM"),
    "RANGE_LOW_VOL":   dict(short_delta=0.18, size_mult=1.00, structure="short strangle / iron fly, premium is thin -> size on vrp"),
    "RANGE_NORMAL":    dict(short_delta=0.15, size_mult=0.90, structure="short strangle with wings"),
    "UNKNOWN":         dict(short_delta=0.08, size_mult=0.25, structure="insufficient history - minimal size"),
}


@dataclass
class Regime:
    label: str
    features: dict
    policy: dict
    trend_side: int = 0            # +1 up-trend, -1 down-trend, 0 none
    notes: list = field(default_factory=list)


def compute_features(df: pd.DataFrame, vix: pd.Series | None = None) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["rv5"] = r.rolling(5).std() * np.sqrt(TRADING_DAYS)
    f["rv20"] = r.rolling(20).std() * np.sqrt(TRADING_DAYS)
    hl = np.log(h / l) ** 2
    f["park10"] = np.sqrt(hl.rolling(10).mean() / (4 * np.log(2))) * np.sqrt(TRADING_DAYS)
    f["vol_ratio"] = f["rv5"] / f["rv20"]
    path = c.diff().abs().rolling(20).sum()
    f["eff_ratio"] = (c - c.shift(20)).abs() / path
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["trend_z"] = (c - c.ewm(span=20, adjust=False).mean()) / atr
    base = (vix.reindex(df.index).ffill() / 100.0) if vix is not None else f["rv20"]
    f["vol_pct"] = base.rolling(252, min_periods=60).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    f["vrp"] = (base - f["rv20"]) if vix is not None else np.nan
    return f


def classify_row(row: pd.Series, cfg: RegimeConfig = RegimeConfig()) -> Regime:
    feats = {k: (None if pd.isna(v) else float(v)) for k, v in row.items() if k != "close"}
    need = ["vol_pct", "vol_ratio", "eff_ratio", "trend_z"]
    if any(feats.get(k) is None for k in need):
        return Regime("UNKNOWN", feats, dict(POLICY["UNKNOWN"]))

    notes = []
    if feats["vol_pct"] >= cfg.stress_vol_pct or feats["vol_ratio"] >= cfg.stress_vol_ratio:
        label = "HIGH_VOL_STRESS"
    elif feats["eff_ratio"] >= cfg.trend_er and abs(feats["trend_z"]) >= cfg.trend_z:
        label = "TRENDING"
    elif feats["vol_pct"] <= cfg.calm_vol_pct and feats["eff_ratio"] <= cfg.calm_er:
        label = "RANGE_LOW_VOL"
    else:
        label = "RANGE_NORMAL"

    policy = dict(POLICY[label])
    vrp = feats.get("vrp")
    if vrp is not None and vrp < 0:
        policy["size_mult"] *= 0.5
        notes.append("negative VRP (implied < realised): size halved")
    side = int(np.sign(feats["trend_z"])) if label == "TRENDING" else 0
    return Regime(label, feats, policy, side, notes)


def classify(df: pd.DataFrame, vix: pd.Series | None = None, cfg: RegimeConfig = RegimeConfig()) -> tuple[Regime, pd.DataFrame]:
    """Return (latest regime, full feature+label history)."""
    f = compute_features(df, vix)
    f["regime"] = [classify_row(r, cfg).label for _, r in f.drop(columns="regime", errors="ignore").iterrows()]
    return classify_row(f.drop(columns="regime").iloc[-1], cfg), f


def main():
    from ..data import load_ohlc_csv, fetch_yahoo, synthetic_ohlc
    ap = argparse.ArgumentParser(description="Classify the current market regime.")
    ap.add_argument("--csv", help="OHLC csv (date,open,high,low,close)")
    ap.add_argument("--vix-csv", help="optional csv with date,close for India VIX")
    ap.add_argument("--yahoo", help="e.g. ^NSEI (needs yfinance)")
    ap.add_argument("--vix-yahoo", help="e.g. ^INDIAVIX")
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    a = ap.parse_args()

    vix = None
    if a.csv:
        df = load_ohlc_csv(a.csv)
        if a.vix_csv:
            vix = pd.read_csv(a.vix_csv, index_col=0, parse_dates=True).iloc[:, -1]
    elif a.yahoo:
        df = fetch_yahoo(a.yahoo)
        vix = fetch_yahoo(a.vix_yahoo)["close"] if a.vix_yahoo else None
    else:
        df, vix = synthetic_ohlc()
        print("[demo] using synthetic data")

    reg, hist = classify(df, vix)
    print(f"Regime: {reg.label}")
    print("Policy:", reg.policy)
    for n in reg.notes:
        print("Note:", n)
    print("Features:", {k: round(v, 3) for k, v in reg.features.items() if v is not None})
    print("\nLast 10 sessions:")
    print(hist[["close", "vol_pct", "vol_ratio", "eff_ratio", "trend_z", "regime"]].tail(10).round(3).to_string())
    print("\nRegime frequency:")
    print(hist["regime"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
