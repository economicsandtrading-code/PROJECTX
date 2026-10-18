"""Black-Scholes(-Merton) pricing, Greeks, implied vol and delta->strike helpers.

Conventions
-----------
* ``opt`` is "C"/"P" (or an array of +1/-1).
* T is in years (use calendar/365 for multi-day, trading-time/252 for intraday work).
* vega is per 1 vol point (1%), theta is per day (``days_per_year`` controls the divisor).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

MIN_T = 1e-8


def _cp(opt):
    if isinstance(opt, str):
        o = opt.upper()
        if o not in ("C", "P", "CE", "PE"):
            raise ValueError(f"opt must be C/P, got {opt!r}")
        return 1.0 if o.startswith("C") else -1.0
    return np.asarray(opt, dtype=float)


def _out(x):
    return float(x) if np.ndim(x) == 0 else x


def d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    T = np.maximum(T, MIN_T)
    sigma = np.maximum(sigma, 1e-8)
    srt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / srt
    return d1, d1 - srt


def bs_price(S, K, T, r, sigma, opt="C", q=0.0):
    cp = _cp(opt)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), MIN_T)
    price = cp * (S * np.exp(-q * T) * norm.cdf(cp * d1) - K * np.exp(-r * T) * norm.cdf(cp * d2))
    return _out(price)


def bs_greeks(S, K, T, r, sigma, opt="C", q=0.0, days_per_year=365.0):
    """Return dict(delta, gamma, vega, theta, rho). vega per vol-pt, theta per day."""
    cp = _cp(opt)
    S, K, sigma = (np.asarray(v, dtype=float) for v in (S, K, sigma))
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), MIN_T)
    sigma = np.maximum(sigma, 1e-8)
    sqrtT = np.sqrt(T)
    fq, disc = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    delta = cp * fq * norm.cdf(cp * d1)
    gamma = fq * pdf / (S * sigma * sqrtT)
    vega = S * fq * pdf * sqrtT / 100.0
    theta_y = (-S * fq * pdf * sigma / (2 * sqrtT)
               - cp * r * K * disc * norm.cdf(cp * d2)
               + cp * q * S * fq * norm.cdf(cp * d1))
    rho = cp * K * T * disc * norm.cdf(cp * d2) / 100.0
    return {k: _out(v) for k, v in
            dict(delta=delta, gamma=gamma, vega=vega, theta=theta_y / days_per_year, rho=rho).items()}


def implied_vol(price, S, K, T, r, opt="C", q=0.0, lo=1e-4, hi=5.0):
    """Scalar implied vol via Brent. Returns nan if price violates no-arbitrage bounds."""
    f = lambda s: bs_price(S, K, T, r, s, opt, q) - price
    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        return float("nan")


def strike_for_delta(S, T, r, sigma, abs_delta, opt="C", q=0.0):
    """Closed-form strike whose |delta| equals ``abs_delta``."""
    cp = _cp(opt)
    sqrtT = np.sqrt(max(T, MIN_T))
    # call: delta = e^{-qT} N(d1);  put: |delta| = e^{-qT} N(-d1)
    d1 = cp * norm.ppf(abs_delta * np.exp(q * T))
    return float(S * np.exp(-d1 * sigma * sqrtT + (r - q + 0.5 * sigma**2) * T))
