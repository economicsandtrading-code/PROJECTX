# options-quant-toolkit

Python toolkit for systematic Indian index-options trading (Nifty / Sensex) plus a corporate-bond screener.
Everything runs offline on sample/synthetic data except the live tracker, which needs Zerodha Kite Connect.

| Module | What it does |
|---|---|
| `quantlab/mtm` | Live mark-to-market P&L tracker (KiteConnect WebSocket) with FIFO realised/unrealised P&L reconstruction |
| `quantlab/regime` | Rule-based market regime classifier + cross-expiry laddering for option selling |
| `quantlab/simulation` | Expiry-day Monte Carlo for Nifty & Sensex (correlated, fat-tailed, path-dependent stops) |
| `quantlab/optimizer` | Cross-index Black-Scholes portfolio optimiser: maximise theta s.t. Greeks / stress limits (MILP) |
| `quantlab/bonds` | Corporate bond screener, YTM via Newton-Raphson (+ duration, DV01, spread) |
| `quantlab/pricing.py` | Shared Black-Scholes price, Greeks, implied vol, delta->strike |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                         # 16 tests
```

## Usage
**1. Live MTM tracker**
```bash
cp .env.example .env              # add KITE_API_KEY / KITE_ACCESS_TOKEN (token expires daily)
export $(grep -v '^#' .env | xargs)
python -m quantlab.mtm.tracker --underlyings NIFTY,SENSEX
# offline replay of a fills file:
python -m quantlab.mtm.tracker --trades-csv examples/sample_trades.csv --ltp-csv examples/sample_ltp.csv
```
Polls `kite.trades()` every 10s, rebuilds the FIFO book (idempotent), subscribes new instrument tokens on
`KiteTicker` in LTP mode, and prints realised / unrealised / fees / net P&L per symbol and in total.
Realised P&L on a closing fill = `sign(lot) * (fill - lot_price) * matched_qty`, so it is correct for shorts.

**2. Regime classifier + ladder**
```bash
python -m quantlab.regime.classifier --demo                       # synthetic data
python -m quantlab.regime.classifier --yahoo ^NSEI --vix-yahoo ^INDIAVIX
python -m quantlab.regime.ladder --spot 24500 --iv 0.14 --dte 2,9,16 --regime TRENDING --trend-side 1
```
Features: realised vol (5d/20d), Parkinson vol, vol ratio, Kaufman efficiency ratio, ATR-scaled stretch,
1-year VIX percentile, VRP proxy. Regimes: `HIGH_VOL_STRESS`, `TRENDING`, `RANGE_LOW_VOL`, `RANGE_NORMAL`
-> each maps to target short delta, size multiplier and structure. The ladder shrinks target delta for later
expiries, weights lots towards near expiries, and pushes the threatened side further OTM in trends.
Thresholds are in `RegimeConfig` - **tune them with walk-forward testing on your own data**.

**3. Expiry-day Monte Carlo**
```bash
python -m quantlab.simulation.expiry_mc --plot pnl.png
python -m quantlab.simulation.expiry_mc --config examples/expiry_config.json
```
Correlated index paths (Cholesky), stochastic path vol + jumps for fat tails, time in trading-years so
`sigma^2 T` matches remaining session variance. Legs are repriced along each path (intrinsic at the close =
cash settlement) so leg-level and portfolio stop-losses are path-dependent. Outputs P&L distribution, win rate,
VaR/CVaR, worst case, short-strike breach probability.
*Known simplifications:* constant IV when repricing; stopped short legs freeze while hedge wings keep running
(this inflates the upside tail - close wings with the shorts if that is not how you trade).

**4. Portfolio optimiser**
```bash
python -m quantlab.optimizer.portfolio --max-loss 60000 --max-vega 20000
```
Net theta, delta, gamma, vega and scenario P&L are linear in position size, so this is solved as an
integer linear program (`scipy.optimize.milp`) - a global optimum, not a local one. Constraints: rupee delta,
short-gamma (1% move), vega, worst-case loss over spot/vol shock scenarios, total lots. Long wings emerge
automatically when protection is cheaper than shrinking the short book. A moneyness skew is applied to IVs.

**5. Bond screener**
```bash
python -m quantlab.bonds.screener data/sample_bonds.csv --settle 2026-10-19 --gsec 6.6 --min-rating A+ --min-spread-bps 150
```
`data/sample_bonds.csv` is **fictional illustrative data**. Supply your own (isin, issuer, rating, coupon %, freq,
maturity, clean price). 30/360 accrual, fractional first period, NR with analytic derivative and a Brent fallback.

## Before you trade anything
* Lot sizes, strike steps and expiry weekdays change - verify `quantlab/config.py` against NSE/BSE circulars.
* Black-Scholes assumes flat vol and lognormal returns; real surfaces have skew and jumps. Treat outputs as risk
  estimates, not guarantees. The optimiser's margin proxy is scenario loss, not SPAN.
* This is research code, not investment advice.

## Layout
```
quantlab/{pricing,config,data}.py
quantlab/mtm/{fifo,tracker}.py
quantlab/regime/{classifier,ladder}.py
quantlab/simulation/expiry_mc.py
quantlab/optimizer/portfolio.py
quantlab/bonds/screener.py
tests/  examples/  data/
```
