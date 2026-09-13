import pandas as pd

from kalshi_edge.market_matcher import evaluate_markets, find_team, parse_matchup
from kalshi_edge.nfl_ratings import TeamRatings


def test_find_team_matches_nickname_and_city():
    assert find_team("Cowboys") == "DAL"
    assert find_team("Dallas Cowboys") == "DAL"
    assert find_team("will the eagles cover?") == "PHI"


def test_find_team_disambiguates_chargers_from_rams():
    assert find_team("LA Chargers") == "LAC"
    assert find_team("LA Rams") == "LA"


def test_parse_matchup_at_separator():
    away, home = parse_matchup("", "Cowboys at Eagles")
    assert (away, home) == ("DAL", "PHI")


def test_parse_matchup_vs_separator():
    away, home = parse_matchup("", "Cowboys vs Eagles")
    assert (away, home) == ("DAL", "PHI")


def _ratings_with_gap():
    r = TeamRatings()
    r.elo["PHI"] = 1600.0
    r.elo["DAL"] = 1500.0
    r.recent_scored["PHI"].extend([27, 24, 30])
    r.recent_allowed["PHI"].extend([17, 20, 14])
    r.recent_scored["DAL"].extend([20, 17, 24])
    r.recent_allowed["DAL"].extend([23, 21, 20])
    return r


def test_evaluate_markets_moneyline_edge():
    ratings = _ratings_with_gap()
    markets = pd.DataFrame([{
        "bet_type": "moneyline",
        "ticker": "KXNFLGAME-25SEP07DALPHI-PHI",
        "event_ticker": "KXNFLGAME-25SEP07DALPHI",
        "title": "Cowboys at Eagles",
        "yes_sub_title": "Eagles",
        "yes_ask": 55,
        "volume": 100,
        "floor_strike": None,
        "cap_strike": None,
    }])
    result = evaluate_markets(markets, ratings)
    assert len(result) == 1
    row = result.iloc[0]
    # Eagles are the stronger (higher Elo) home team, so model should favor
    # them above a 55c market price -> positive edge.
    assert row["model_prob"] > 0.55
    assert row["edge"] > 0


def test_evaluate_markets_skips_unparseable_rows():
    ratings = _ratings_with_gap()
    markets = pd.DataFrame([{
        "bet_type": "moneyline",
        "ticker": "weird-ticker",
        "event_ticker": "weird-ticker",
        "title": "Some unparseable title",
        "yes_sub_title": "???",
        "yes_ask": 55,
        "volume": 100,
        "floor_strike": None,
        "cap_strike": None,
    }])
    result = evaluate_markets(markets, ratings)
    assert result.empty


def test_evaluate_markets_total_over():
    ratings = _ratings_with_gap()
    markets = pd.DataFrame([{
        "bet_type": "total",
        "ticker": "KXNFLTOTAL-25SEP07DALPHI-40.5",
        "event_ticker": "KXNFLGAME-25SEP07DALPHI",
        "title": "Cowboys at Eagles total over 40.5",
        "yes_sub_title": None,
        "yes_ask": 40,
        "volume": 50,
        "floor_strike": 40.5,
        "cap_strike": None,
    }])
    result = evaluate_markets(markets, ratings)
    assert len(result) == 1
    assert result.iloc[0]["side"] == "Over 40.5"
