"""Central configuration. Verify contract specs against the latest NSE/BSE circulars
before trading -- lot sizes and expiry weekdays are revised periodically."""

RISK_FREE_RATE = 0.06          # annualised, continuous; adjust to current T-bill / MIBOR
DIVIDEND_YIELD = 0.0

# lot_size and strike_step per underlying (CHECK EXCHANGE CIRCULARS - these change)
INDEX_SPECS = {
    "NIFTY":  {"lot_size": 65, "strike_step": 50},
    "SENSEX": {"lot_size": 20, "strike_step": 100},
}

TRADING_DAYS = 252
SESSION_MINUTES = 375          # 09:15 - 15:30 IST
