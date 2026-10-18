"""Corporate bond screener: yield-to-maturity via Newton-Raphson.

Pricing equation (per 100 face, periodic compounding f times a year, 30/360 accrual):

    dirty = sum_{k=0}^{n-1} CF_k / (1 + y/f)^(w + k)        w = days(settle->next coupon)/days(coupon period)

    f(y)  = PV(y) - dirty
    f'(y) = - sum (w+k)/f * CF_k / (1 + y/f)^(w+k+1)
    y_{n+1} = y_n - f(y_n)/f'(y_n)        (falls back to Brent if NR leaves the bracket / fails)

Also computes accrued interest, Macaulay/modified duration, DV01 and spread over a benchmark.

Input CSV columns: isin,issuer,rating,coupon,freq,maturity,clean_price     (coupon in % p.a.)
"""
from __future__ import annotations

import argparse
import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from scipy.optimize import brentq

RATING_SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "C", "D"]
_RATING_RE = re.compile(r"^(AAA|AA|A|BBB|BB|B|C|D)([+-])?")
_AGENCIES = re.compile(r"CRISIL|ICRA|CARE|ACUITE|BRICKWORK|INDRA|INDIARATINGS|IND|FITCH|SP|S&P")


def parse_rating(s: str) -> int:
    """Rank on RATING_SCALE (0 = AAA). Tolerates 'CRISIL AA+', '[ICRA]AA-', 'AAA(CE)', 'IND AA'."""
    s = str(s).upper()
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", s)        # drop [ICRA] / (CE) style tags
    s = _AGENCIES.sub("", s).replace(" ", "")
    m = _RATING_RE.match(s)
    if not m:
        return len(RATING_SCALE)
    key = m.group(1) + (m.group(2) or "")
    return RATING_SCALE.index(key) if key in RATING_SCALE else RATING_SCALE.index(m.group(1))


def days360(d1: date, d2: date) -> int:
    a, b = min(d1.day, 30), d2.day
    if b == 31 and a == 30:
        b = 30
    return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (b - a)


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def coupon_schedule(settle: date, maturity: date, freq: int) -> tuple[date, list[date]]:
    """(previous coupon date, remaining coupon dates > settle)."""
    step = 12 // freq
    dates, i = [], 0
    d = maturity
    while d > settle:
        dates.append(d)
        i += 1
        d = add_months(maturity, -step * i)
    return d, dates[::-1]


@dataclass
class BondResult:
    ytm: float
    accrued: float
    dirty: float
    mac_duration: float
    mod_duration: float
    dv01: float
    current_yield: float
    iterations: int
    method: str


def analyse(clean: float, coupon_pct: float, maturity: date, settle: date, freq: int = 2,
            face: float = 100.0, tol: float = 1e-12, max_iter: int = 100) -> BondResult:
    prev, sched = coupon_schedule(settle, maturity, freq)
    if not sched:
        raise ValueError("bond already matured")
    cpn = face * coupon_pct / 100.0 / freq
    n = len(sched)
    cfs = [cpn] * n
    cfs[-1] += face
    period = days360(prev, sched[0])
    to_next = days360(settle, sched[0])
    w = to_next / period
    accrued = cpn * (period - to_next) / period
    dirty = clean + accrued
    t = [w + k for k in range(n)]

    def pv(y):
        return sum(cf / (1 + y / freq) ** tk for cf, tk in zip(cfs, t))

    def dpv(y):
        return sum(-tk / freq * cf / (1 + y / freq) ** (tk + 1) for cf, tk in zip(cfs, t))

    y = max(coupon_pct / 100.0, 0.01) * face / clean          # initial guess ~ current yield
    method, it = "newton", 0
    for it in range(1, max_iter + 1):
        fy, dfy = pv(y) - dirty, dpv(y)
        if dfy == 0:
            break
        y_new = y - fy / dfy
        if y_new <= -freq * 0.99:                              # left the domain of (1 + y/f) > 0
            break
        if abs(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    else:
        pass
    if abs(pv(y) - dirty) > 1e-8:                              # robust fallback
        method = "brent"
        y = brentq(lambda v: pv(v) - dirty, -freq * 0.99, 5.0, xtol=1e-14)

    price_parts = [cf / (1 + y / freq) ** tk for cf, tk in zip(cfs, t)]
    mac = sum((tk / freq) * p for tk, p in zip(t, price_parts)) / sum(price_parts)
    mod = mac / (1 + y / freq)
    return BondResult(ytm=y, accrued=accrued, dirty=dirty, mac_duration=mac, mod_duration=mod,
                      dv01=mod * dirty * 1e-4, current_yield=coupon_pct / clean, iterations=it, method=method)


def screen(df: pd.DataFrame, settle: date, gsec_yield: float = 0.07, min_rating: str = "A",
           min_ytm: float | None = None, min_spread_bps: float | None = None,
           min_years: float = 0.0, max_years: float = 50.0, max_mod_dur: float | None = None) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        mat = pd.to_datetime(r.maturity).date()
        if mat <= settle:
            continue
        res = analyse(float(r.clean_price), float(r.coupon), mat, settle, int(r.freq))
        rows.append(dict(isin=r.isin, issuer=r.issuer, rating=r.rating, coupon=r.coupon, maturity=mat,
                         years=round((mat - settle).days / 365.25, 2), clean=r.clean_price,
                         accrued=round(res.accrued, 3), ytm_pct=round(res.ytm * 100, 3),
                         spread_bps=round((res.ytm - gsec_yield) * 1e4, 1), mod_dur=round(res.mod_duration, 3),
                         dv01=round(res.dv01, 4), cur_yld_pct=round(res.current_yield * 100, 3),
                         iters=res.iterations, _rank=parse_rating(r.rating)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    m = (out["_rank"] <= parse_rating(min_rating)) & out["years"].between(min_years, max_years)
    if min_ytm is not None:
        m &= out["ytm_pct"] >= min_ytm
    if min_spread_bps is not None:
        m &= out["spread_bps"] >= min_spread_bps
    if max_mod_dur is not None:
        m &= out["mod_dur"] <= max_mod_dur
    return out[m].sort_values("ytm_pct", ascending=False).drop(columns="_rank").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Corporate bond screener (YTM via Newton-Raphson).")
    ap.add_argument("csv", nargs="?", default="data/sample_bonds.csv")
    ap.add_argument("--settle", default=date.today().isoformat())
    ap.add_argument("--gsec", type=float, default=7.0, help="benchmark G-sec yield in %%")
    ap.add_argument("--min-rating", default="A")
    ap.add_argument("--min-ytm", type=float)
    ap.add_argument("--min-spread-bps", type=float)
    ap.add_argument("--min-years", type=float, default=0.0)
    ap.add_argument("--max-years", type=float, default=50.0)
    ap.add_argument("--max-mod-dur", type=float)
    ap.add_argument("--out", help="write screened results to CSV")
    a = ap.parse_args()
    settle = datetime.strptime(a.settle, "%Y-%m-%d").date()
    df = pd.read_csv(a.csv)
    res = screen(df, settle, a.gsec / 100, a.min_rating, a.min_ytm, a.min_spread_bps,
                 a.min_years, a.max_years, a.max_mod_dur)
    print(f"settlement {settle} | benchmark {a.gsec}% | {len(res)}/{len(df)} bonds pass\n")
    print(res.to_string(index=False) if len(res) else "no bonds pass the filters")
    if a.out:
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
