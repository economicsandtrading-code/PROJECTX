from datetime import date
import numpy as np
from quantlab.bonds.screener import analyse, parse_rating, coupon_schedule


def test_par_bond_ytm_equals_coupon_on_coupon_date():
    r = analyse(100.0, 8.0, date(2031, 3, 15), date(2026, 3, 15), freq=2)
    assert abs(r.ytm - 0.08) < 1e-9 and abs(r.accrued) < 1e-9


def test_price_roundtrip_and_duration():
    r = analyse(96.5, 7.5, date(2030, 1, 10), date(2026, 10, 19), freq=1)
    # reprice at YTM and compare to dirty price
    prev, sched = coupon_schedule(date(2026, 10, 19), date(2030, 1, 10), 1)
    from quantlab.bonds.screener import days360
    w = days360(date(2026, 10, 19), sched[0]) / days360(prev, sched[0])
    cfs = [7.5] * len(sched); cfs[-1] += 100
    pv = sum(cf / (1 + r.ytm) ** (w + k) for k, cf in enumerate(cfs))
    assert abs(pv - r.dirty) < 1e-7
    assert 0 < r.mod_duration < r.mac_duration < 4
    # DV01 vs bump-and-reprice
    pv_up = sum(cf / (1 + r.ytm + 1e-4) ** (w + k) for k, cf in enumerate(cfs))
    assert abs((r.dirty - pv_up) - r.dv01) < 5e-4


def test_zero_coupon():
    r = analyse(80.0, 0.0, date(2029, 1, 1), date(2026, 1, 1), freq=1)
    assert abs(r.ytm - ((100 / 80) ** (1 / 3) - 1)) < 1e-6


def test_ratings():
    assert parse_rating("CRISIL AA+") < parse_rating("[ICRA]AA-") < parse_rating("A")
    assert parse_rating("AAA(CE)") == 0
