import numpy as np
from quantlab.data import synthetic_ohlc
from quantlab.regime.classifier import classify, POLICY
from quantlab.regime.ladder import build_ladder
from quantlab.simulation.expiry_mc import IndexSpec, Leg, simulate_paths, evaluate
from quantlab.optimizer.portfolio import Underlying, Limits, build_candidates, optimise


def test_regime_runs_and_labels_valid():
    df, vix = synthetic_ohlc(600)
    reg, hist = classify(df, vix)
    assert reg.label in POLICY
    assert set(hist["regime"]) <= set(POLICY)


def test_ladder_short_only_and_delta_decay():
    df, vix = synthetic_ohlc(600)
    reg, _ = classify(df, vix)
    lad = build_ladder(24500, reg, [2, 9, 16], 0.14, base_lots=12)
    assert (lad["lots"] < 0).all()
    puts = lad[lad.opt == "P"]["delta"].abs().to_numpy()
    assert puts[0] >= puts[-1]


def test_mc_martingale_and_correlation():
    sp = {"NIFTY": IndexSpec("NIFTY", 24500, 0.14, 65), "SENSEX": IndexSpec("SENSEX", 80500, 0.14, 20)}
    p = simulate_paths(sp, corr=0.95, n_paths=60000, seed=1)
    for k, s in sp.items():
        assert abs(p[k][:, -1].mean() / s.spot - 1) < 0.002
    c = np.corrcoef(np.log(p["NIFTY"][:, -1]), np.log(p["SENSEX"][:, -1]))[0, 1]
    assert 0.90 < c < 0.99


def test_mc_short_strangle_edge_and_stops_cap_loss():
    sp = {"NIFTY": IndexSpec("NIFTY", 24500, 0.14, 65)}
    legs = [Leg("NIFTY", "C", 24700, -2), Leg("NIFTY", "P", 24300, -2)]
    p = simulate_paths(sp, n_paths=20000, seed=3)
    no_sl = evaluate(legs, sp, p)
    sl = evaluate(legs, sp, p, portfolio_sl_inr=15000, cost_per_lot=0)
    assert sl["stats"]["worst"] >= -15000 * 1.02 - 1
    assert no_sl["stats"]["worst"] < sl["stats"]["worst"]


def test_optimizer_respects_limits():
    lim = Limits(max_scenario_loss_inr=40000, max_abs_vega_inr=10000, max_total_lots=40)
    cands = build_candidates([Underlying("NIFTY", 24500, [2, 9], 0.14), Underlying("SENSEX", 80500, [2, 9], 0.14)], limits=lim)
    pos, s = optimise(cands, lim, time_limit=20)
    assert s["net_theta_inr_day"] > 0
    assert abs(s["net_vega_inr"]) <= lim.max_abs_vega_inr + 1e-6
    assert abs(s["net_delta_inr"]) <= lim.max_abs_delta_inr + 1e-6
    assert s["net_gamma_1pct_inr"] >= -lim.max_short_gamma_inr - 1e-6
    assert s["worst_scenario_pnl_inr"] >= -lim.max_scenario_loss_inr - 1e-6
    assert s["total_lots"] <= lim.max_total_lots
