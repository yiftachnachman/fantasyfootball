from kalshi_edge.kelly import kelly_fraction, kelly_stake


def test_kelly_fraction_zero_at_fair_price():
    # price == true probability -> no edge -> f* should be ~0
    f = kelly_fraction(prob=0.60, price_cents=60)
    assert abs(f) < 1e-9


def test_kelly_fraction_positive_when_underpriced():
    # model says 65% but market prices it at 55c -> positive edge
    f = kelly_fraction(prob=0.65, price_cents=55)
    assert f > 0


def test_kelly_fraction_negative_when_overpriced():
    f = kelly_fraction(prob=0.45, price_cents=55)
    assert f < 0


def test_kelly_stake_is_zero_with_no_edge():
    stake = kelly_stake(prob=0.50, price_cents=50, bankroll=1000)
    assert stake == 0.0


def test_kelly_stake_respects_max_fraction_cap():
    # huge apparent edge should still be capped
    stake = kelly_stake(
        prob=0.99, price_cents=5, bankroll=1000,
        kelly_multiplier=1.0, max_stake_fraction=0.05,
    )
    assert stake == 1000 * 0.05


def test_kelly_stake_scales_with_bankroll():
    small = kelly_stake(prob=0.65, price_cents=55, bankroll=100)
    large = kelly_stake(prob=0.65, price_cents=55, bankroll=1000)
    assert large == small * 10
