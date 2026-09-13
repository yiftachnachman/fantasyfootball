import pandas as pd

from kalshi_edge.market_matcher import (
    evaluate_markets,
    find_all_teams,
    opponent_from_event_code,
    parse_event_date,
    split_matchup_code,
    team_from_ticker_suffix,
)
from kalshi_edge.nfl_ratings import TeamRatings, normalize_team


def test_team_from_ticker_suffix_moneyline():
    assert team_from_ticker_suffix("KXNFLGAME-26SEP20INDKC-KC") == "KC"
    assert team_from_ticker_suffix("KXNFLGAME-26SEP21NYGLAR-LAR") == "LAR"


def test_team_from_ticker_suffix_spread_strips_trailing_digits():
    assert team_from_ticker_suffix("KXNFLSPREAD-26SEP17DETBUF-DET8") == "DET"


def test_parse_event_date():
    d = parse_event_date("KXNFLGAME-26SEP20INDKC")
    assert (d.year, d.month, d.day) == (2026, 9, 20)


def test_opponent_from_event_code_suffix_match():
    assert opponent_from_event_code("KXNFLGAME-26SEP20INDKC", "KC") == "IND"


def test_opponent_from_event_code_prefix_match():
    assert opponent_from_event_code("KXNFLSPREAD-26SEP17DETBUF", "DET") == "BUF"


def test_split_matchup_code_unambiguous():
    assert split_matchup_code("MIASF") == ("MIA", "SF")
    assert split_matchup_code("LVLAC") == ("LV", "LAC")


def test_find_all_teams_order_and_dedup():
    teams = find_all_teams("If New York G wins the NY Giants vs LA Rams Pro Football game")
    assert teams == ["NYG", "LA"]


def _ratings_with_gap_and_schedule():
    r = TeamRatings()
    r.elo["PHI"] = 1600.0
    r.elo["DAL"] = 1500.0
    r.recent_scored["PHI"].extend([27, 24, 30])
    r.recent_allowed["PHI"].extend([17, 20, 14])
    r.recent_scored["DAL"].extend([20, 17, 24])
    r.recent_allowed["DAL"].extend([23, 21, 20])
    # PHI hosts DAL -- this is what resolve_home_away should surface.
    r.matchups[frozenset({"PHI", "DAL"})] = [
        (pd.Timestamp("2026-09-21"), "PHI", "DAL"),
    ]
    return r


def test_evaluate_markets_moneyline_edge():
    ratings = _ratings_with_gap_and_schedule()
    markets = pd.DataFrame([{
        "bet_type": "moneyline",
        "ticker": "KXNFLGAME-26SEP21DALPHI-PHI",
        "event_ticker": "KXNFLGAME-26SEP21DALPHI",
        "title": "Philadelphia wins",
        "yes_ask_dollars": "0.5500",
        "volume_fp": "100.00",
        "floor_strike": None,
        "cap_strike": None,
        "rules_primary": "If Philadelphia wins the Dallas vs Philadelphia game...",
    }])
    result = evaluate_markets(markets, ratings)
    assert len(result) == 1
    row = result.iloc[0]
    # Eagles are the stronger (higher Elo) home team, so model should favor
    # them above a 55c market price -> positive edge.
    assert row["model_prob"] > 0.55
    assert row["edge"] > 0


def test_evaluate_markets_skips_rows_with_no_schedule_match():
    ratings = TeamRatings()  # no matchups loaded
    markets = pd.DataFrame([{
        "bet_type": "moneyline",
        "ticker": "KXNFLGAME-26SEP21DALPHI-PHI",
        "event_ticker": "KXNFLGAME-26SEP21DALPHI",
        "title": "Philadelphia wins",
        "yes_ask_dollars": "0.5500",
        "volume_fp": "100.00",
        "floor_strike": None,
        "cap_strike": None,
        "rules_primary": "If Philadelphia wins the Dallas vs Philadelphia game...",
    }])
    result = evaluate_markets(markets, ratings)
    assert result.empty


def test_evaluate_markets_spread_uses_ticker_team_and_line():
    ratings = _ratings_with_gap_and_schedule()
    markets = pd.DataFrame([{
        "bet_type": "spread",
        "ticker": "KXNFLSPREAD-26SEP21DALPHI-PHI4",
        "event_ticker": "KXNFLSPREAD-26SEP21DALPHI",
        "title": "Philadelphia wins by over 3.5 points?",
        "yes_ask_dollars": "0.5000",
        "volume_fp": "50.00",
        "floor_strike": 3.5,
        "cap_strike": None,
        "rules_primary": "If Philadelphia wins by more than 3.5 points in the Dallas vs Philadelphia game...",
    }])
    result = evaluate_markets(markets, ratings)
    assert len(result) == 1
    assert result.iloc[0]["side"] == "PHI covers +3.5"


def test_evaluate_markets_total_over():
    ratings = _ratings_with_gap_and_schedule()
    markets = pd.DataFrame([{
        "bet_type": "total",
        "ticker": "KXNFLTOTAL-26SEP21DALPHI-40",
        "event_ticker": "KXNFLTOTAL-26SEP21DALPHI",
        "title": "Total points over 40.5?",
        "yes_ask_dollars": "0.4000",
        "volume_fp": "50.00",
        "floor_strike": 40.5,
        "cap_strike": None,
        "rules_primary": "If the total score of the Dallas vs Philadelphia game is over 40.5...",
    }])
    result = evaluate_markets(markets, ratings)
    assert len(result) == 1
    assert result.iloc[0]["side"] == "Over 40.5"
