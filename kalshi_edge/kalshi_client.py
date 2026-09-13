"""Kalshi NFL market data puller.

Pulls current NFL game contracts across all three bet types from Kalshi's
PUBLIC market-data endpoint -- no API key needed for this part. You'll only
need a Kalshi key later, once you're ready to place orders.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

# Both api.elections.kalshi.com and external-api.kalshi.com show up in
# current Kalshi docs/clients for the same v2 API. If this one 404s for
# you, swap in the other -- confirm against docs.kalshi.com.
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# NFL game-level series tickers. KXNFLWINS (season win totals) is a
# separate season-long futures ladder, not a per-game market, so it's
# left out here.
NFL_SERIES = {
    "moneyline": "KXNFLGAME",
    "spread": "KXNFLSPREAD",
    "total": "KXNFLTOTAL",
}

MARKET_FIELDS = (
    "ticker",
    "event_ticker",
    "series_ticker",
    "title",
    "subtitle",
    "yes_sub_title",
    "no_sub_title",
    "rules_primary",
    # Kalshi returns prices as decimal-dollar strings, e.g. "0.2400" for
    # 24 cents -- NOT as the plain cents-integer fields ("yes_bid",
    # "yes_ask") an older API version used. See market_matcher.py for the
    # dollars-string -> cents conversion.
    "yes_bid_dollars",
    "yes_ask_dollars",
    "no_bid_dollars",
    "no_ask_dollars",
    "last_price_dollars",
    "volume_fp",
    "open_interest_fp",
    "floor_strike",
    "cap_strike",
    "strike_type",
    "close_time",
    "status",
)


def get_markets(series_ticker: str, status: str = "open") -> list[dict[str, Any]]:
    """Pull every market for a series ticker, paging through results."""
    markets: list[dict[str, Any]] = []
    cursor = None
    while True:
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)  # stay polite re: rate limits
    return markets


def markets_to_df(markets: list[dict[str, Any]], bet_type: str) -> pd.DataFrame:
    """Flatten raw market dicts into a tidy frame, keeping only fields the
    edge-matching logic needs (see market_matcher.py for how these are used).

    Prices come back as decimal-dollar strings (e.g. yes_ask_dollars =
    "0.6300"), which double as the implied probability of the "yes" side --
    it costs 63 cents to buy a contract that pays $1 if "yes" happens, i.e.
    the market prices that side at roughly 63%.
    """
    rows = [{field: m.get(field) for field in MARKET_FIELDS} for m in markets]
    df = pd.DataFrame(rows, columns=list(MARKET_FIELDS))
    df.insert(0, "bet_type", bet_type)
    return df


def pull_all_nfl_markets(status: str = "open") -> pd.DataFrame:
    frames = [
        markets_to_df(get_markets(series, status=status), bet_type)
        for bet_type, series in NFL_SERIES.items()
    ]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = pull_all_nfl_markets()
    print(f"Pulled {len(df)} open NFL contracts across {df['bet_type'].nunique()} bet types\n")
    print(df.sort_values(["bet_type", "close_time"]).to_string(index=False))
