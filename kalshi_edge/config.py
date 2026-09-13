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
    elo_home_field: float = 39.0  # ~1.6 points at 25 Elo/point -- fit against
    # Vegas closing spread_line over 2023-2025 (see backtest.py); zeroes out
    # a small (+0.63 pt) mean bias found there, nothing more dramatic.
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

    # --- situational adjustments (QB status, other injuries, weather) ---
    # A QB change (current depth-chart QB1 differs from who started the
    # season, or the current QB1 is reported Out/Doubtful) is worth several
    # points of spread -- this is a rough, undifferentiated estimate (a
    # change to a good backup costs less than to a bad one; we can't tell
    # those apart here), not fit against data the way elo_home_field was.
    qb_change_elo_penalty: float = 100.0  # ~4 pts at 25 elo/point
    qb_questionable_elo_penalty: float = 40.0  # ~1.6 pts -- partial risk, most Questionables play
    key_injury_elo_penalty_each: float = 6.0  # ~0.24 pts per non-QB "Out" starter
    key_injury_elo_penalty_cap: float = 30.0  # ~1.2 pts max from non-QB injuries combined

    # Weather (outdoor stadiums only -- see situational.DOME_TEAMS), applied
    # to predicted_total only. Coefficients are rough literature priors
    # (wind hurts passing/scoring; heavy precip and cold do too), not fit.
    weather_wind_threshold_mph: float = 15.0
    weather_wind_coeff: float = 0.15  # total points removed per mph over threshold
    weather_precip_threshold_mm: float = 2.0
    weather_precip_total_penalty: float = 3.0
    weather_cold_threshold_f: float = 32.0
    weather_cold_coeff: float = 0.05  # total points removed per degree under threshold
    weather_total_delta_cap: float = 8.0  # max combined weather reduction to predicted_total


SETTINGS = Settings()
