"""Team power ratings built from historical NFL results (via nflreadpy).

Two complementary ratings come out of this:

- Elo ratings -> win probability and predicted point margin (good for the
  moneyline and spread markets).
- Rolling offense/defense scoring averages -> predicted game total (good
  for the totals market). Elo alone doesn't tell you if a game is a 17-14
  slog or a 34-31 shootout; two teams can have the same Elo gap with very
  different expected totals.

Nothing here is fit/optimized against a holdout set -- the constants in
config.py are reasonable priors, not the result of a grid search. Backtest
before trusting this with real money (see README).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

from .config import SETTINGS

# nflreadpy/nflverse team abbreviations drift over the years (relocations,
# rebrands). Normalize to the current abbreviation so history stitches
# together into one continuous rating per franchise.
TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "WSH": "WAS",
    "JAC": "JAX",
}


def normalize_team(code: str) -> str:
    code = (code or "").upper().strip()
    return TEAM_ALIASES.get(code, code)


def _load_schedules(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    schedules = nfl.load_schedules(seasons=seasons)
    # nflreadpy returns a polars DataFrame; convert if needed so the rest
    # of this module can stay in plain pandas.
    if hasattr(schedules, "to_pandas"):
        schedules = schedules.to_pandas()
    return schedules


@dataclass
class TeamRatings:
    elo: dict[str, float] = field(default_factory=dict)
    recent_scored: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=SETTINGS.scoring_lookback_games)))
    recent_allowed: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=SETTINGS.scoring_lookback_games)))
    league_avg_points: float = 22.0
    # Kalshi's market data never says which team is home -- only the two
    # teams and the game date. This maps {team_a, team_b} -> every
    # scheduled meeting between them (gameday, home_team, away_team),
    # played or not, so market_matcher can look up the real home/away
    # assignment instead of guessing from ticker/title text order.
    matchups: dict[frozenset, list[tuple]] = field(default_factory=dict)

    def resolve_home_away(self, team_a: str, team_b: str, near_date=None) -> tuple[str, str] | None:
        """Return (home_team, away_team) for the meeting between team_a
        and team_b closest to near_date (or the only/most recent meeting
        if near_date is None). None if they don't appear to have played
        or be scheduled to play each other in the loaded seasons.
        """
        key = frozenset({normalize_team(team_a), normalize_team(team_b)})
        candidates = self.matchups.get(key)
        if not candidates:
            return None
        if near_date is None:
            gameday, home, away = candidates[-1]
        else:
            gameday, home, away = min(candidates, key=lambda c: abs((c[0] - near_date).days))
        return home, away

    def get_elo(self, team: str) -> float:
        return self.elo.get(normalize_team(team), SETTINGS.elo_initial)

    def off_avg(self, team: str) -> float:
        history = self.recent_scored.get(normalize_team(team))
        return sum(history) / len(history) if history else self.league_avg_points

    def def_avg(self, team: str) -> float:
        history = self.recent_allowed.get(normalize_team(team))
        return sum(history) / len(history) if history else self.league_avg_points

    def predict_game(self, home_team: str, away_team: str) -> dict[str, float]:
        """Return model predictions for a single matchup.

        - home_win_prob / away_win_prob: from Elo, including home field.
        - predicted_margin: home_score - away_score, expected value.
        - predicted_total: home_score + away_score, expected value.
        """
        home, away = normalize_team(home_team), normalize_team(away_team)
        elo_diff = (self.get_elo(home) + SETTINGS.elo_home_field) - self.get_elo(away)
        home_win_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        predicted_margin = elo_diff / SETTINGS.elo_points_per_elo

        # Predicted total blends each side's offense against the other's
        # defense -- a simple, transparent stand-in for a real scoring model.
        home_expected = (self.off_avg(home) + self.def_avg(away)) / 2
        away_expected = (self.off_avg(away) + self.def_avg(home)) / 2
        predicted_total = home_expected + away_expected

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": 1.0 - home_win_prob,
            "predicted_margin": predicted_margin,  # home - away
            "predicted_total": predicted_total,
        }


def _mov_multiplier(elo_diff_pre: float, margin: float) -> float:
    """538-style margin-of-victory multiplier so blowouts move Elo more
    than narrow wins, while diminishing returns kick in against big
    pre-game favorites (avoids over-crediting a favorite for beating up
    on a team it was already expected to blow out).
    """
    if not SETTINGS.elo_mov_multiplier:
        return 1.0
    return ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * abs(elo_diff_pre))


def build_ratings(seasons: list[int] | None = None) -> TeamRatings:
    """Walk completed games in chronological order, updating Elo and
    rolling scoring averages after each result.
    """
    if seasons is None:
        current_year = pd.Timestamp.utcnow().year
        seasons = list(range(current_year - SETTINGS.seasons_of_history + 1, current_year + 1))

    schedules = _load_schedules(seasons)
    schedules = schedules.copy()
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    ratings = TeamRatings()

    # Every scheduled matchup (played or not) feeds the home/away lookup --
    # future games already have home_team/away_team assigned even before
    # they're played.
    matchups: dict[frozenset, list[tuple]] = defaultdict(list)
    for row in schedules.itertuples():
        home, away = normalize_team(row.home_team), normalize_team(row.away_team)
        matchups[frozenset({home, away})].append((row.gameday, home, away))
    ratings.matchups = dict(matchups)

    played = schedules.dropna(subset=["home_score", "away_score"]).copy()
    played = played.sort_values(["gameday", "week"])

    all_scores = pd.concat([played["home_score"], played["away_score"]])
    if len(all_scores):
        ratings.league_avg_points = float(all_scores.mean())

    for row in played.itertuples():
        home, away = normalize_team(row.home_team), normalize_team(row.away_team)
        home_score, away_score = float(row.home_score), float(row.away_score)

        home_elo = ratings.elo.get(home, SETTINGS.elo_initial)
        away_elo = ratings.elo.get(away, SETTINGS.elo_initial)

        elo_diff_pre = (home_elo + SETTINGS.elo_home_field) - away_elo
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff_pre / 400.0))
        actual_home = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)

        margin = home_score - away_score
        k = SETTINGS.elo_k * _mov_multiplier(elo_diff_pre, margin)
        shift = k * (actual_home - expected_home)

        ratings.elo[home] = home_elo + shift
        ratings.elo[away] = away_elo - shift

        ratings.recent_scored[home].append(home_score)
        ratings.recent_allowed[home].append(away_score)
        ratings.recent_scored[away].append(away_score)
        ratings.recent_allowed[away].append(home_score)

    return ratings
