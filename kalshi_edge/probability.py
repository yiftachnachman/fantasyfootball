"""Turn point predictions into cover/over probabilities.

We model final margin and total as normally distributed around the point
prediction. The sigma constants (config.margin_sigma / total_sigma) are
the well-documented rough std deviations of NFL margin and total around a
good pre-game line -- see README for sourcing notes.
"""

from __future__ import annotations

import math

from .config import SETTINGS


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cover_probability(predicted_margin_for_team: float, line: float, sigma: float = SETTINGS.margin_sigma) -> float:
    """P(team's actual margin > line).

    `predicted_margin_for_team` and `line` must use the same sign
    convention: positive means that team is favored/expected to win by
    that many points. E.g. team favored by 6.5 (line = -6.5, i.e. must
    win by more than 6.5 to cover) with a model margin of +9 covers with
    cover_probability(9, 6.5).
    """
    z = (predicted_margin_for_team - line) / sigma
    return _norm_cdf(z)


def over_probability(predicted_total: float, line: float, sigma: float = SETTINGS.total_sigma) -> float:
    """P(actual total > line)."""
    z = (predicted_total - line) / sigma
    return _norm_cdf(z)


def price_to_prob(price_cents: float) -> float:
    """Kalshi cent price IS the market's implied probability."""
    return price_cents / 100.0
