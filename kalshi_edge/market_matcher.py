"""Map raw Kalshi market rows to model predictions and compute edge.

Verified against a live pull of KXNFLGAME / KXNFLSPREAD / KXNFLTOTAL
(2026-09-13). Kalshi's title/subtitle text turned out to name only ONE
team ("Kansas City wins", "Detroit wins by over 7.5 points?") with no
"X at Y" pattern, and never states which team is home -- so instead of
parsing prose we lean on ticker structure, which is precise:

    KXNFLGAME-26SEP20INDKC-KC        moneyline: "KC" = the team this
                                      contract is about
    KXNFLSPREAD-26SEP17DETBUF-DET8   spread: "DET" (strip trailing digits)
    KXNFLTOTAL-26SEP17DETBUF-72      total: no team, just a line

The event ticker's middle segment is `<7-char date><team codes
concatenated>` (e.g. "26SEP20" + "INDKC"). Since we already know the
market's own team from the final ticker segment, we recover the
opponent by stripping that code from either end of the concatenated
pair -- no guessing needed. Kalshi never exposes which side is home, so
home/away is resolved by matching {team_a, team_b} + the ticker's date
against the real NFL schedule (nfl_ratings.TeamRatings.resolve_home_away).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .nfl_ratings import TeamRatings, normalize_team
from .probability import cover_probability, over_probability, price_to_prob

# Exact abbreviations Kalshi uses in ticker codes (JAC not JAX, LAR/LAC
# both present as distinct 3-letter codes, etc.) -- used only to split an
# ambiguous concatenated pair like "MIASF" for total markets, which have
# no per-team ticker suffix to anchor on.
KALSHI_TICKER_CODES = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAC", "KC", "LV", "LAC", "LAR", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
})

# Fallback keyword matching over rules_primary text (e.g. "NY Giants vs LA
# Rams"), used only when ticker-code arithmetic can't resolve a matchup.
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
_TEAM_NAME_KEYS = sorted(TEAM_NAMES.keys(), key=len, reverse=True)

_LINE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_LEADING_LETTERS_RE = re.compile(r"^[A-Za-z]+")
_DATE_RE = re.compile(r"^(\d{2}[A-Za-z]{3}\d{2})")


def find_team(text: str) -> str | None:
    """First team keyword found in text, or None. Used only as a fallback
    when ticker-code arithmetic doesn't resolve (see module docstring)."""
    if not text:
        return None
    lowered = text.lower()
    for key in _TEAM_NAME_KEYS:
        if key in lowered:
            return TEAM_NAMES[key]
    return None


def find_all_teams(text: str) -> list[str]:
    """Every distinct team keyword found in text, in order of first
    appearance (e.g. "NY Giants vs LA Rams" -> ["NYG", "LA"])."""
    if not text:
        return []
    lowered = text.lower()
    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for key in _TEAM_NAME_KEYS:
        idx = lowered.find(key)
        if idx != -1:
            code = TEAM_NAMES[key]
            if code not in seen:
                hits.append((idx, code))
                seen.add(code)
    hits.sort(key=lambda h: h[0])
    return [code for _, code in hits]


def team_from_ticker_suffix(ticker: str) -> str | None:
    """The team a moneyline/spread contract is about, from the ticker's
    final segment: "...-KC" -> "KC", "...-DET8" -> "DET"."""
    if not ticker or "-" not in ticker:
        return None
    suffix = ticker.rsplit("-", 1)[-1]
    match = _LEADING_LETTERS_RE.match(suffix)
    return match.group().upper() if match else None


def parse_event_date(event_ticker: str) -> pd.Timestamp | None:
    """"KXNFLGAME-26SEP20INDKC" -> Timestamp('2026-09-20')."""
    if not event_ticker or "-" not in event_ticker:
        return None
    code = event_ticker.split("-", 1)[1]
    match = _DATE_RE.match(code)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%y%b%d")
    except ValueError:
        return None


def _event_team_code_blob(event_ticker: str) -> str | None:
    """The concatenated team-codes portion of an event ticker, e.g.
    "KXNFLGAME-26SEP20INDKC" -> "INDKC" (date prefix stripped)."""
    if not event_ticker or "-" not in event_ticker:
        return None
    code = event_ticker.split("-", 1)[1]
    match = _DATE_RE.match(code)
    if not match:
        return None
    return code[len(match.group(1)):]


def opponent_from_event_code(event_ticker: str, own_team: str) -> str | None:
    """Given the market's own team (from the ticker suffix), recover the
    opponent by stripping own_team from either end of the event ticker's
    concatenated team-code blob."""
    blob = _event_team_code_blob(event_ticker)
    if not blob or not own_team:
        return None
    if blob.endswith(own_team):
        remainder = blob[: -len(own_team)]
    elif blob.startswith(own_team):
        remainder = blob[len(own_team):]
    else:
        return None
    return remainder or None


def split_matchup_code(blob: str) -> tuple[str, str] | None:
    """Split a concatenated team-code blob (e.g. "MIASF") into two known
    Kalshi ticker codes, for total markets which have no team suffix to
    anchor on. Returns None if the split is ambiguous or invalid."""
    valid_splits = [
        (blob[:i], blob[i:])
        for i in (2, 3)
        if blob[:i] in KALSHI_TICKER_CODES and blob[i:] in KALSHI_TICKER_CODES
    ]
    if len(valid_splits) == 1:
        return valid_splits[0]
    return None


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


def _yes_ask_cents(m: dict) -> float | None:
    raw = m.get("yes_ask_dollars")
    if raw is None:
        return None
    try:
        return float(raw) * 100.0
    except (TypeError, ValueError):
        return None


def _matchup_teams(m: dict, own_team: str | None) -> tuple[str, str] | None:
    """Best-effort (team_a, team_b) for this market's game, order
    unspecified. Tries ticker-code arithmetic first, falls back to
    rules_primary text."""
    event_ticker = m.get("event_ticker", "")
    if own_team:
        opponent = opponent_from_event_code(event_ticker, own_team)
        if opponent and opponent in KALSHI_TICKER_CODES:
            return own_team, opponent

    blob = _event_team_code_blob(event_ticker)
    if blob:
        split = split_matchup_code(blob)
        if split:
            return split

    teams = find_all_teams(m.get("rules_primary") or m.get("title") or "")
    if len(teams) >= 2:
        return teams[0], teams[1]
    return None


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


def _resolve_home_away_for_market(m: dict, ratings: TeamRatings, event_date) -> tuple[str, str, str | None] | None:
    """Returns (home, away, own_team) for any bet_type. own_team is None
    for total markets, which have no single team the contract is about."""
    bet_type = m.get("bet_type")
    if bet_type in ("moneyline", "spread"):
        own_team = team_from_ticker_suffix(m.get("ticker", ""))
        teams = _matchup_teams(m, own_team)
        if not (own_team and teams):
            return None
        opponent = teams[1] if normalize_team(teams[0]) == normalize_team(own_team) else teams[0]
        home_away = ratings.resolve_home_away(own_team, opponent, near_date=event_date)
        if home_away is None:
            return None
        return home_away[0], home_away[1], own_team

    if bet_type == "total":
        teams = _matchup_teams(m, None)
        if not teams:
            return None
        home_away = ratings.resolve_home_away(teams[0], teams[1], near_date=event_date)
        if home_away is None:
            return None
        return home_away[0], home_away[1], None

    return None


def evaluate_markets(markets_df: pd.DataFrame, ratings: TeamRatings, context=None) -> pd.DataFrame:
    """Compute model probability, market-implied probability, and edge for
    every market row we can resolve to a real matchup. Rows we can't (no
    schedule match, unrecognized ticker shape, ...) are dropped from the
    output but not from the input frame -- run with --dump-unmatched to
    inspect them.

    `context`, if given a situational.SituationalContext, layers QB/injury
    and weather adjustments onto the base Elo/scoring-average prediction
    (see situational.py). Omit it (the default) to use the base model only.
    """
    results: list[EdgeRow] = []

    for m in markets_df.to_dict("records"):
        bet_type = m.get("bet_type")
        title = m.get("title") or ""
        market_prob = None
        yes_ask_cents = _yes_ask_cents(m)
        if yes_ask_cents is not None:
            market_prob = price_to_prob(yes_ask_cents)
        if market_prob is None:
            continue

        event_date = parse_event_date(m.get("event_ticker", ""))
        resolved = _resolve_home_away_for_market(m, ratings, event_date)
        if resolved is None:
            continue
        home, away, own_team = resolved
        pred = ratings.predict_game(home, away)

        if context is not None:
            from .situational import adjust_prediction

            week = context.week_for_date(event_date)
            home_penalty = context.qb_penalty_elo(home, week) + context.key_injury_penalty_elo(home, week)
            away_penalty = context.qb_penalty_elo(away, week) + context.key_injury_penalty_elo(away, week)
            total_delta = context.weather_total_delta(home, event_date)
            pred = adjust_prediction(pred, home_penalty, away_penalty, total_delta)

        if bet_type == "moneyline":
            model_prob = pred["home_win_prob"] if normalize_team(own_team) == normalize_team(home) else pred["away_win_prob"]
            side = f"{own_team} to win"
        elif bet_type == "spread":
            line = _line_from_market(m)
            if line is None:
                continue
            team_margin = pred["predicted_margin"] if normalize_team(own_team) == normalize_team(home) else -pred["predicted_margin"]
            model_prob = cover_probability(team_margin, line)
            side = f"{own_team} covers {line:+.1f}"
        elif bet_type == "total":
            line = _line_from_market(m)
            if line is None:
                continue
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
            yes_ask=yes_ask_cents,
            volume=int(float(m.get("volume_fp") or 0)),
            edge=model_prob - market_prob,
        ))

    return pd.DataFrame([r.__dict__ for r in results])


def unmatched_reason(m: dict, ratings: TeamRatings) -> str | None:
    """Why evaluate_markets would skip this row, or None if it wouldn't be."""
    bet_type = m.get("bet_type")
    if _yes_ask_cents(m) is None:
        return "no yes_ask_dollars"

    if bet_type in ("moneyline", "spread"):
        own_team = team_from_ticker_suffix(m.get("ticker", ""))
        if not own_team:
            return "couldn't extract own team from ticker suffix"
        teams = _matchup_teams(m, own_team)
        if not teams:
            return "couldn't determine opponent"
        opponent = teams[1] if normalize_team(teams[0]) == normalize_team(own_team) else teams[0]
        event_date = parse_event_date(m.get("event_ticker", ""))
        if ratings.resolve_home_away(own_team, opponent, near_date=event_date) is None:
            return f"no schedule match for {own_team} vs {opponent}"
        if bet_type == "spread" and _line_from_market(m) is None:
            return "no spread line found"
        return None

    if bet_type == "total":
        teams = _matchup_teams(m, None)
        if not teams:
            return "couldn't determine matchup teams"
        event_date = parse_event_date(m.get("event_ticker", ""))
        if ratings.resolve_home_away(teams[0], teams[1], near_date=event_date) is None:
            return f"no schedule match for {teams[0]} vs {teams[1]}"
        if _line_from_market(m) is None:
            return "no total line found"
        return None

    return f"unrecognized bet_type {bet_type!r}"
