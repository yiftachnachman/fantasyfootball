"""Kelly-criterion stake sizing for a binary Yes/No contract.

Buying a YES contract at price c (dollars, 0 < c < 1) costs c and pays $1
if it resolves Yes, $0 otherwise. Net odds b = (1 - c) / c. Kelly fraction
for a bet with true win probability p:

    f* = p - (1 - p) / b = p - (1 - p) * c / (1 - c)

We apply a fractional-Kelly multiplier (config.kelly_multiplier) and a
hard cap (config.max_stake_fraction) because full Kelly is extremely
volatile against a model whose "true probability" is itself an estimate.
"""

from __future__ import annotations

from .config import SETTINGS


def kelly_fraction(prob: float, price_cents: float) -> float:
    c = price_cents / 100.0
    if c <= 0.0 or c >= 1.0:
        return 0.0
    f = prob - (1.0 - prob) * c / (1.0 - c)
    return f


def kelly_stake(
    prob: float,
    price_cents: float,
    bankroll: float = SETTINGS.bankroll,
    kelly_multiplier: float = SETTINGS.kelly_multiplier,
    max_stake_fraction: float = SETTINGS.max_stake_fraction,
) -> float:
    """Dollar stake to risk on this contract, 0 if the raw Kelly fraction
    is non-positive (i.e. no edge or negative edge)."""
    f = kelly_fraction(prob, price_cents)
    if f <= 0:
        return 0.0
    fraction = min(f * kelly_multiplier, max_stake_fraction)
    return bankroll * fraction
