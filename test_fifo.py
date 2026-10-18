from quantlab.mtm.fifo import Fill, FifoBook


def mk(side_qty, price, sym="NIFTY26OCT24500CE"):
    return Fill(ts=0, symbol=sym, qty=side_qty, price=price)


def test_short_then_cover_fifo():
    b = FifoBook()
    b.add_fill(mk(-130, 90))
    b.add_fill(mk(-65, 70))
    b.add_fill(mk(+100, 60))                # closes 100 of the 90-lot first (FIFO)
    assert b.realized["NIFTY26OCT24500CE"] == 100 * (90 - 60)
    assert b.net_qty("NIFTY26OCT24500CE") == -95
    assert abs(b.avg_price("NIFTY26OCT24500CE") - (30 * 90 + 65 * 70) / 95) < 1e-9


def test_long_close_and_flip():
    b = FifoBook()
    b.add_fill(mk(+65, 50))
    b.add_fill(mk(-130, 80))                # closes long, opens 65 short
    assert b.realized["NIFTY26OCT24500CE"] == 65 * 30
    assert b.net_qty("NIFTY26OCT24500CE") == -65
    assert b.unrealized("NIFTY26OCT24500CE", 70) == -65 * (70 - 80)


def test_total_pnl_equals_cash_flow():
    b = FifoBook()
    fills = [mk(-65, 100), mk(+65, 80), mk(-130, 60), mk(+65, 40), mk(+65, 45)]
    for f in fills:
        b.add_fill(f)
    cash = -sum(f.qty * f.price for f in fills)         # flat at the end -> realised == cash
    assert b.net_qty("NIFTY26OCT24500CE") == 0
    assert abs(b.realized["NIFTY26OCT24500CE"] - cash) < 1e-9
