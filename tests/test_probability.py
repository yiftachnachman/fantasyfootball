import math

from kalshi_edge.probability import cover_probability, over_probability, price_to_prob


def test_price_to_prob():
    assert price_to_prob(63) == 0.63
    assert price_to_prob(1) == 0.01
    assert price_to_prob(99) == 0.99


def test_cover_probability_at_the_line_is_half():
    # predicted margin exactly equal to the line -> coin flip
    assert math.isclose(cover_probability(3.5, 3.5, sigma=13.86), 0.5, abs_tol=1e-9)


def test_cover_probability_increases_with_predicted_margin():
    low = cover_probability(0.0, 3.5, sigma=13.86)
    high = cover_probability(10.0, 3.5, sigma=13.86)
    assert high > low


def test_over_probability_at_the_line_is_half():
    assert math.isclose(over_probability(45.0, 45.0, sigma=10.5), 0.5, abs_tol=1e-9)


def test_over_probability_bounds():
    assert 0.0 < over_probability(30.0, 45.0, sigma=10.5) < 0.5
    assert 0.5 < over_probability(60.0, 45.0, sigma=10.5) < 1.0
