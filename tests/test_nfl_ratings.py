import pandas as pd

from kalshi_edge.config import SETTINGS
from kalshi_edge.nfl_ratings import TeamRatings, _regress_toward_mean, normalize_team


def test_normalize_team_applies_known_aliases():
    assert normalize_team("JAC") == "JAX"
    assert normalize_team("LAR") == "LA"
    assert normalize_team("OAK") == "LV"


def test_normalize_team_passes_through_unknown_codes():
    assert normalize_team("KC") == "KC"


def test_get_elo_defaults_to_initial_for_unseen_team():
    r = TeamRatings()
    assert r.get_elo("KC") == SETTINGS.elo_initial


def test_off_def_avg_default_to_league_average_with_no_history():
    r = TeamRatings(league_avg_points=21.5)
    assert r.off_avg("KC") == 21.5
    assert r.def_avg("KC") == 21.5


def test_predict_game_favors_higher_elo_home_team():
    r = TeamRatings()
    r.elo["KC"] = 1600.0
    r.elo["DEN"] = 1500.0
    pred = r.predict_game("KC", "DEN")
    assert pred["home_win_prob"] > 0.5
    assert pred["predicted_margin"] > 0
    assert abs(pred["home_win_prob"] + pred["away_win_prob"] - 1.0) < 1e-9


def test_predict_game_equal_elo_home_field_favors_home():
    r = TeamRatings()
    r.elo["KC"] = 1500.0
    r.elo["DEN"] = 1500.0
    pred = r.predict_game("KC", "DEN")
    # equal Elo -- home field advantage alone should push win prob above 50%
    assert pred["home_win_prob"] > 0.5


def test_resolve_home_away_picks_closest_date():
    r = TeamRatings()
    r.matchups[frozenset({"KC", "DEN"})] = [
        (pd.Timestamp("2024-09-01"), "KC", "DEN"),
        (pd.Timestamp("2024-12-01"), "DEN", "KC"),
    ]
    home, away = r.resolve_home_away("KC", "DEN", near_date=pd.Timestamp("2024-11-28"))
    assert (home, away) == ("DEN", "KC")


def test_resolve_home_away_none_when_no_matchup_recorded():
    r = TeamRatings()
    assert r.resolve_home_away("KC", "DEN") is None


def test_regress_toward_mean_pulls_toward_initial():
    r = TeamRatings()
    r.elo["KC"] = 1700.0
    r.elo["DEN"] = 1300.0
    _regress_toward_mean(r)
    frac = SETTINGS.elo_season_regression
    assert r.elo["KC"] == SETTINGS.elo_initial + (1700.0 - SETTINGS.elo_initial) * (1 - frac)
    assert r.elo["DEN"] == SETTINGS.elo_initial + (1300.0 - SETTINGS.elo_initial) * (1 - frac)
    # regression pulls both closer to the initial rating, never past it
    assert SETTINGS.elo_initial < r.elo["KC"] < 1700.0
    assert 1300.0 < r.elo["DEN"] < SETTINGS.elo_initial


def test_regress_toward_mean_is_a_no_op_when_disabled():
    import dataclasses
    import kalshi_edge.nfl_ratings as nfl_ratings

    original_settings = nfl_ratings.SETTINGS
    try:
        nfl_ratings.SETTINGS = dataclasses.replace(original_settings, elo_season_regression=0.0)
        r = TeamRatings()
        r.elo["KC"] = 1700.0
        nfl_ratings._regress_toward_mean(r)
        assert r.elo["KC"] == 1700.0
    finally:
        nfl_ratings.SETTINGS = original_settings
