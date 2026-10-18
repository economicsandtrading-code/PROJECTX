"""Data helpers: CSV loaders, optional Yahoo download, and a synthetic regime-switching generator
(used by the demos and tests so nothing needs network access)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df[["open", "high", "low", "close"]].astype(float)


def fetch_yahoo(ticker: str, period: str = "3y") -> pd.DataFrame:
    import yfinance as yf  # optional dependency
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close"]].dropna()


def synthetic_ohlc(n: int = 900, seed: int = 7, s0: float = 22000.0) -> tuple[pd.DataFrame, pd.Series]:
    """Markov regime-switching GBM -> (OHLC, synthetic VIX series in index points)."""
    rng = np.random.default_rng(seed)
    # regimes: calm range, normal, trending-up, stress
    vols = np.array([0.09, 0.14, 0.13, 0.30])
    drifts = np.array([0.02, 0.08, 0.45, -0.60])
    P = np.array([[0.97, 0.03, 0.00, 0.00],
                  [0.02, 0.95, 0.025, 0.005],
                  [0.00, 0.06, 0.93, 0.01],
                  [0.00, 0.10, 0.00, 0.90]])
    state, rets, st = 1, [], []
    for _ in range(n):
        state = rng.choice(4, p=P[state])
        st.append(state)
        rets.append(drifts[state] / 252 + vols[state] / np.sqrt(252) * rng.standard_normal())
    rets = np.array(rets)
    close = s0 * np.exp(np.cumsum(rets))
    prev = np.r_[s0, close[:-1]]
    rng_pct = np.abs(rets) + vols[st] / np.sqrt(252) * np.abs(rng.standard_normal(n)) * 0.6
    high = np.maximum(prev, close) * (1 + 0.5 * rng_pct)
    low = np.minimum(prev, close) * (1 - 0.5 * rng_pct)
    idx = pd.date_range(start="2023-01-02", periods=n, freq="B")
    df = pd.DataFrame(dict(open=prev, high=high, low=low, close=close), index=idx)
    vix = pd.Series(vols[st] * 100 * (1.1 + 0.1 * rng.standard_normal(n)), index=idx, name="vix").clip(lower=7)
    return df, vix
