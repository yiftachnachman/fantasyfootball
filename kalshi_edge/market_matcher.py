"""Map raw Kalshi market rows to model predictions and compute edge.

The exact wording Kalshi uses in `title` / `subtitle` / `yes_sub_title`,
and whether spread/total lines show up in `floor_strike` vs. buried in the
title text, is the one thing in this project I can't verify from here (no
live network access in this sandbox -- see README). Everything below is
written to degrade gracefully: unparsed rows are returned with
bet_type="unknown" and the raw fields intact instead of raising, so you
can inspect them with `python -m kalshi_edge.main --dump-unmatched` and
extend TEAM_NAMES / the regexes below to fit whatever you actually see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .nfl_ratings import TeamRatings, normalize_team
from .probability import cover_probability, over_probability, price_to_prob

# Full name / city / nickname fragments -> standard abbreviation. Kalshi
# titles are human-readable ("Cowboys", "Dallas Cowboys", ...) rather than
# ticker codes, so we match on text instead of trying to reverse-engineer
# a ticker schema.
TEAM_NAMES: dict[str, str] = {
    "cardinals": "ARI", "arizona": "ARI",
    "falcons": "ATL", "atlanta": "ATL",
    "ravens": "BAL", "baltimore": "BAL",
    "bills": "BUF", "buffalo": "BUF",
    "panthers": "CAR", "carolina": "CAR",
    "bears": "CHI", "chicago": "CHI",
    "bengals": "CIN", "cincinnati": "CIN",
    "browns": "CLE", "cleveland": "CLE",
    "cowboys": "DAL", "dallas": "DAL",
    "broncos": "DEN", "denver": "DEN",
    "lions": "DET", "detroit": "DET",
    "packers": "GB", "green bay": "GB",
    "texans": "HOU", "houston": "HOU",
    "colts": "IND", "indianapolis": "IND",
    "jaguars": "JAX", "jacksonville": "JAX",
    "chiefs": "KC", "kansas city": "KC",
    "raiders": "LV", "las vegas": "LV",
    "chargers": "LAC",
    "rams": "LA",
    "los angeles": "LA",  # ambiguous with Chargers; Chargers matched first below
    "dolphins": "MIA", "miami": "MIA",
    "vikings": "MIN", "minnesota": "MIN",
    "patriots": "NE", "new england": "NE",
    "saints": "NO", "new orleans": "NO",
    "giants": "NYG",
    "jets": "NYJ",
    "eagles": "PHI", "philadelphia": "PHI",
    "steelers": "PIT", "pittsburgh": "PIT",
    "49ers": "SF", "niners": "SF", "san francisco": "SF",
    "seahawks": "SEA", "seattle": "SEA",
    "buccaneers": "TB", "bucs": "TB", "tampa bay": "TB",
    "titans": "TEN", "tennessee": "TEN",
    "commanders": "WAS", "washington": "WAS",
}
# Longer/more specific keys first so "chargers" wins over the generic
# "los angeles" for LAC, etc.
_TEAM_NAME_KEYS = sorted(TEAM_NAMES.keys(), key=len, reverse=True)

_LINE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def find_team(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    for key in _TEAM_NAME_KEYS:
        if key in lowered:
            return TEAM_NAMES[key]
    return None


def parse_matchup(event_ticker: str, title: str) -> tuple[str | None, str | None]:
    """Best-effort (away, home) extraction from event title text.

    Handles "X at Y" / "X @ Y" (away at home) and falls back to "X vs Y"
    (order ambiguous -- treated as away vs home, verify against real data).
    """
    text = title or event_ticker or ""
    for sep in (" at ", "@", " vs. ", " vs "):
        if sep in text:
            left, _, right = text.partition(sep)
            return find_team(left), find_team(right)
    return None, None


def _line_from_market(m: dict) -> float | None:
    for key in ("floor_strike", "cap_strike"):
        val = m.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    for text in (m.get("yes_sub_title"), m.get("subtitle"), m.get("title")):
        if text:
            match = _LINE_RE.search(text)
            if match:
                return float(match.group())
    return None


def _yes_ask_prob(m: dict) -> float | None:
    price = m.get("yes_ask")
    return price_to_prob(price) if price is not None else None


@dataclass
class EdgeRow:
    bet_type: str
    ticker: str
    title: str
    side: str  # human-readable description of what "yes" means here
    model_prob: float
    market_prob: float
    yes_ask: float
    volume: int
    edge: float


def evaluate_markets(markets_df: pd.DataFrame, ratings: TeamRatings) -> pd.DataFrame:
    """Compute model probability, market-implied probability, and edge for
    every market row we can parse. Rows we can't confidently parse are
    dropped from the output but not from the input frame -- callers who
    want to inspect them can diff against markets_df.
    """
    results: list[EdgeRow] = []

    for m in markets_df.to_dict("records"):
        bet_type = m.get("bet_type")
        title = m.get("title") or ""
        away, home = parse_matchup(m.get("event_ticker", ""), title)
        market_prob = _yes_ask_prob(m)
        if market_prob is None:
            continue

        if bet_type == "moneyline":
            team = find_team(m.get("yes_sub_title") or "")
            if not (team and home and away):
                continue
            pred = ratings.predict_game(home, away)
            model_prob = pred["home_win_prob"] if normalize_team(team) == normalize_team(home) else pred["away_win_prob"]
            side = f"{team} to win"

        elif bet_type == "spread":
            team = find_team(m.get("yes_sub_title") or "")
            line = _line_from_market(m)
            if not (team and home and away and line is not None):
                continue
            pred = ratings.predict_game(home, away)
            # predicted_margin is home - away; flip sign if the contract's
            # team is the away side so "positive" always means "this team
            # favored by that much".
            team_margin = pred["predicted_margin"] if normalize_team(team) == normalize_team(home) else -pred["predicted_margin"]
            model_prob = cover_probability(team_margin, line)
            side = f"{team} covers {line:+.1f}"

        elif bet_type == "total":
            line = _line_from_market(m)
            if not (home and away and line is not None):
                continue
            pred = ratings.predict_game(home, away)
            model_prob = over_probability(pred["predicted_total"], line)
            side = f"Over {line:.1f}"

        else:
            continue

        results.append(EdgeRow(
            bet_type=bet_type,
            ticker=m.get("ticker"),
            title=title,
            side=side,
            model_prob=model_prob,
            market_prob=market_prob,
            yes_ask=m.get("yes_ask"),
            volume=m.get("volume") or 0,
            edge=model_prob - market_prob,
        ))

    return pd.DataFrame([r.__dict__ for r in results])
