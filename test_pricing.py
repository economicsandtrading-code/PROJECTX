import numpy as np
from quantlab.pricing import bs_price, bs_greeks, implied_vol, strike_for_delta


def test_put_call_parity():
    S, K, T, r, q, s = 24500, 24600, 0.05, 0.06, 0.01, 0.15
    c, p = bs_price(S, K, T, r, s, "C", q), bs_price(S, K, T, r, s, "P", q)
    assert abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-8


def test_greeks_match_finite_difference():
    S, K, T, r, s = 24500, 24300, 0.04, 0.06, 0.14
    g = bs_greeks(S, K, T, r, s, "P")
    h = 1.0
    d = (bs_price(S + h, K, T, r, s, "P") - bs_price(S - h, K, T, r, s, "P")) / (2 * h)
    gm = (bs_price(S + h, K, T, r, s, "P") - 2 * bs_price(S, K, T, r, s, "P") + bs_price(S - h, K, T, r, s, "P")) / h**2
    v = (bs_price(S, K, T, r, s + 1e-4, "P") - bs_price(S, K, T, r, s - 1e-4, "P")) / 2e-4 / 100
    h_t = 1e-6
    th = -(bs_price(S, K, T + h_t, r, s, "P") - bs_price(S, K, T - h_t, r, s, "P")) / (2 * h_t) / 365
    assert abs(g["delta"] - d) < 1e-5 and abs(g["gamma"] - gm) < 1e-6
    assert abs(g["vega"] - v) < 1e-4 and abs(g["theta"] - th) < 1e-3


def test_implied_vol_roundtrip():
    px = bs_price(24500, 24800, 0.08, 0.06, 0.173, "C")
    assert abs(implied_vol(px, 24500, 24800, 0.08, 0.06, "C") - 0.173) < 1e-7


def test_strike_for_delta():
    for opt in "CP":
        K = strike_for_delta(24500, 0.05, 0.06, 0.14, 0.15, opt)
        assert abs(abs(bs_greeks(24500, K, 0.05, 0.06, 0.14, opt)["delta"]) - 0.15) < 1e-9
