"""Tunable knobs for the model and staking logic.

Nothing here is holy writ -- these are reasonable starting points pulled
from public NFL forecasting literature (538's NFL Elo writeups, standard
point-spread/total variance estimates). Recalibrate against your own
backtests before sizing real money.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # --- Elo model ---
    elo_initial: float = 1500.0
    elo_k: float = 20.0
    elo_home_field: float = 55.0  # ~2.2 points at 25 Elo/point
    elo_points_per_elo: float = 25.0  # conversion: Elo diff / this = point spread
    elo_mov_multiplier: bool = True  # scale K by how lopsided the result was

    # --- history window ---
    seasons_of_history: int = 4  # trailing seasons used to seed/update ratings
    scoring_lookback_games: int = 8  # rolling window for off/def scoring averages

    # --- outcome variance (points), used to turn a predicted margin/total
    # into a cover/over probability via the normal CDF ---
    margin_sigma: float = 13.86
    total_sigma: float = 10.5

    # --- staking ---
    bankroll: float = 1000.0
    kelly_multiplier: float = 0.25  # fractional Kelly; full Kelly is too hot
    max_stake_fraction: float = 0.05  # hard cap per position regardless of Kelly
    min_edge: float = 0.03  # skip anything modeled inside +/-3 points of probability
    min_volume: int = 0  # set > 0 to filter out illiquid contracts

    seed_bankroll_currency: str = "USD"


SETTINGS = Settings()
