import pandas as pd

from kalshi_edge.situational import SituationalContext, adjust_prediction
from kalshi_edge.config import SETTINGS


def test_adjust_prediction_no_penalties_is_a_no_op():
    # home_win_prob is always recomputed from predicted_margin (via the
    # same logistic transform used to derive it originally), so the
    # fixture must be internally consistent -- 3.0 pts at the default
    # 25 elo/point implies this exact win probability.
    elo_diff = 3.0 * SETTINGS.elo_points_per_elo
    home_win_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    pred = {"home_win_prob": home_win_prob, "away_win_prob": 1 - home_win_prob, "predicted_margin": 3.0, "predicted_total": 45.0}
    adjusted = adjust_prediction(pred, home_penalty_elo=0.0, away_penalty_elo=0.0, total_delta=0.0)
    assert adjusted["predicted_margin"] == pred["predicted_margin"]
    assert abs(adjusted["home_win_prob"] - pred["home_win_prob"]) < 1e-9
    assert adjusted["predicted_total"] == pred["predicted_total"]


def test_adjust_prediction_home_penalty_reduces_home_win_prob():
    pred = {"home_win_prob": 0.6, "away_win_prob": 0.4, "predicted_margin": 3.0, "predicted_total": 45.0}
    adjusted = adjust_prediction(pred, home_penalty_elo=100.0, away_penalty_elo=0.0, total_delta=0.0)
    assert adjusted["home_win_prob"] < pred["home_win_prob"]
    assert adjusted["predicted_margin"] < pred["predicted_margin"]


def test_adjust_prediction_away_penalty_helps_home():
    pred = {"home_win_prob": 0.5, "away_win_prob": 0.5, "predicted_margin": 0.0, "predicted_total": 45.0}
    adjusted = adjust_prediction(pred, home_penalty_elo=0.0, away_penalty_elo=100.0, total_delta=0.0)
    assert adjusted["home_win_prob"] > 0.5
    assert adjusted["predicted_margin"] > 0.0


def test_adjust_prediction_total_delta_only_moves_total():
    pred = {"home_win_prob": 0.5, "away_win_prob": 0.5, "predicted_margin": 0.0, "predicted_total": 45.0}
    adjusted = adjust_prediction(pred, home_penalty_elo=0.0, away_penalty_elo=0.0, total_delta=-4.0)
    assert adjusted["predicted_total"] == 41.0
    assert adjusted["home_win_prob"] == 0.5


def test_qb_penalty_zero_when_no_change_and_no_injury_flag():
    ctx = SituationalContext(
        baseline_qb1={"MIA": "qb-a"},
        current_qb1={"MIA": "qb-a"},
    )
    assert ctx.qb_penalty_elo("MIA") == 0.0


def test_qb_penalty_full_when_starter_changed():
    ctx = SituationalContext(
        baseline_qb1={"MIA": "qb-a"},
        current_qb1={"MIA": "qb-b"},
    )
    assert ctx.qb_penalty_elo("MIA") == SETTINGS.qb_change_elo_penalty


def test_qb_penalty_partial_when_current_starter_questionable():
    ctx = SituationalContext(
        baseline_qb1={"MIA": "qb-a"},
        current_qb1={"MIA": "qb-a"},
        injury_status_by_week={3: {("MIA", "qb-a"): "Questionable"}},
    )
    assert ctx.qb_penalty_elo("MIA", week=3) == SETTINGS.qb_questionable_elo_penalty


def test_qb_penalty_full_when_current_starter_out():
    ctx = SituationalContext(
        baseline_qb1={"MIA": "qb-a"},
        current_qb1={"MIA": "qb-a"},
        injury_status_by_week={3: {("MIA", "qb-a"): "Out"}},
    )
    assert ctx.qb_penalty_elo("MIA", week=3) == SETTINGS.qb_change_elo_penalty


def test_qb_penalty_ignores_injury_report_from_a_different_week():
    # week 1's "Out" designation shouldn't apply to a week-3 market --
    # the report may be stale (game already played) or simply not about
    # this game.
    ctx = SituationalContext(
        baseline_qb1={"MIA": "qb-a"},
        current_qb1={"MIA": "qb-a"},
        injury_status_by_week={1: {("MIA", "qb-a"): "Out"}},
    )
    assert ctx.qb_penalty_elo("MIA", week=3) == 0.0


def test_qb_penalty_skips_injury_tier_when_week_is_none():
    ctx = SituationalContext(
        baseline_qb1={"MIA": "qb-a"},
        current_qb1={"MIA": "qb-a"},
        injury_status_by_week={3: {("MIA", "qb-a"): "Out"}},
    )
    assert ctx.qb_penalty_elo("MIA", week=None) == 0.0


def test_week_for_date_matches_nearest_schedule_week():
    ctx = SituationalContext(_week_dates=[
        (pd.Timestamp("2026-09-13"), 1),
        (pd.Timestamp("2026-09-20"), 2),
        (pd.Timestamp("2026-09-27"), 3),
    ])
    assert ctx.week_for_date(pd.Timestamp("2026-09-21")) == 2


def test_week_for_date_none_when_too_far_from_any_game():
    ctx = SituationalContext(_week_dates=[(pd.Timestamp("2026-09-13"), 1)])
    assert ctx.week_for_date(pd.Timestamp("2026-12-01")) is None


def test_week_for_date_none_without_event_date_or_schedule():
    ctx = SituationalContext()
    assert ctx.week_for_date(None) is None
    assert ctx.week_for_date(pd.Timestamp("2026-09-20")) is None


def test_qb_penalty_normalizes_team_code():
    # "LAR" and "LA" should be treated as the same team (see nfl_ratings.TEAM_ALIASES)
    ctx = SituationalContext(baseline_qb1={"LA": "qb-a"}, current_qb1={"LA": "qb-b"})
    assert ctx.qb_penalty_elo("LAR") == SETTINGS.qb_change_elo_penalty


def test_key_injury_penalty_counts_out_non_qb_starters_and_caps():
    ctx = SituationalContext(
        current_qb1={"MIA": "qb-a"},
        injury_status_by_week={3: {
            ("MIA", "qb-a"): "Out",  # the QB itself -- must not be double-counted here
            ("MIA", "wr-1"): "Out",
            ("MIA", "wr-2"): "Out",
            ("MIA", "te-1"): "Questionable",  # not "Out" -- doesn't count
        }},
    )
    assert ctx.key_injury_penalty_elo("MIA", week=3) == 2 * SETTINGS.key_injury_elo_penalty_each


def test_key_injury_penalty_caps_at_configured_max():
    injuries = {("MIA", f"p{i}"): "Out" for i in range(20)}
    ctx = SituationalContext(current_qb1={}, injury_status_by_week={3: injuries})
    assert ctx.key_injury_penalty_elo("MIA", week=3) == SETTINGS.key_injury_elo_penalty_cap


def test_key_injury_penalty_zero_when_week_is_none():
    ctx = SituationalContext(injury_status_by_week={3: {("MIA", "wr-1"): "Out"}})
    assert ctx.key_injury_penalty_elo("MIA", week=None) == 0.0


def test_weather_total_delta_zero_for_dome_team_no_network_call():
    ctx = SituationalContext()
    # ARI is in DOME_TEAMS -- must short-circuit before any weather fetch
    assert ctx.weather_total_delta("ARI", pd.Timestamp("2026-09-20")) == 0.0


def test_weather_total_delta_zero_when_date_missing():
    ctx = SituationalContext()
    assert ctx.weather_total_delta("BUF", None) == 0.0


def test_weather_total_delta_uses_cached_forecast():
    ctx = SituationalContext()
    lat, lon = (42.7738, -78.7870)  # BUF
    date = "2026-09-20"
    ctx._weather_cache[(lat, lon, date)] = {
        "temp_min": 20.0,  # well under the cold threshold
        "windspeed_max": 30.0,  # well over the wind threshold
        "precipitation_sum": 5.0,  # over the precip threshold
    }
    delta = ctx.weather_total_delta("BUF", pd.Timestamp(date))
    assert delta < 0.0
    assert delta >= -SETTINGS.weather_total_delta_cap
