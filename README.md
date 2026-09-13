# fantasyfootball

An edge-finding tool for Kalshi's NFL contracts: moneyline (`KXNFLGAME`),
spread (`KXNFLSPREAD`), and total (`KXNFLTOTAL`). It builds a simple
power-rating model from historical results, converts that into a win /
cover / over probability for each open Kalshi contract, and ranks
contracts by edge (model probability minus market-implied probability).

Kalshi prices are already probabilities: a `yes_ask` of 63 means it costs
63 cents for a contract that pays $1 if "yes" happens, i.e. the market is
pricing that outcome at ~63%. No odds conversion needed — the model's job
is just to produce a better-calibrated probability than that.

## How it works

```
kalshi_edge/
  kalshi_client.py   pulls open markets from Kalshi's public market-data API
  nfl_ratings.py      builds Elo + rolling scoring-average ratings from nflreadpy history
  probability.py      turns a predicted margin/total into a cover/over probability
  market_matcher.py    matches Kalshi market text to a matchup + line, computes edge
  kelly.py             fractional-Kelly stake sizing
  config.py            every tunable constant lives here
  main.py               CLI that wires it all together
```

Two ratings come out of the model:

- **Elo** (with home-field advantage and a margin-of-victory multiplier,
  in the style of 538's NFL Elo) drives win probability and the predicted
  point margin, which feeds the spread market.
- **Rolling offense/defense scoring averages** (last 8 games) drive the
  predicted total, since two teams can have the same Elo gap but very
  different expected combined scoring.

Predicted margin/total are treated as normally distributed (sigma ≈ 13.9
points for margin, ≈ 10.5 for totals — standard NFL variance estimates)
and converted to a cover/over probability via the normal CDF.

Edge = model probability − market probability. Stake sizing uses
fractional Kelly (25% of full Kelly by default, hard-capped at 5% of
bankroll per position) — see `kelly.py`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
python -m kalshi_edge.main
# tune thresholds:
python -m kalshi_edge.main --min-edge 0.05 --bankroll 500 --kelly-multiplier 0.25
```

This prints every contract that clears `--min-edge` and writes the full
ranked list (all matched contracts, not just the ones clearing the
threshold) to `nfl_edges.csv`.

If a market can't be matched to a team/line (see "Known limitation"
below), it's silently skipped in the ranked output. To see what got
skipped and why:

```bash
python -m kalshi_edge.main --dump-unmatched
```

## Tests

The model math (probability conversion, Kelly sizing, text parsing) is
covered by unit tests with no network dependency:

```bash
pytest
```

## Known limitation — verify market parsing against live data

I could not reach `api.elections.kalshi.com` from the sandbox that wrote
this code (network policy blocks it), so `market_matcher.py` was written
against my best understanding of Kalshi's market schema, not verified
against a live response. Specifically:

- Team names are matched by searching `title` / `event_ticker` /
  `yes_sub_title` text for city/nickname keywords (see `TEAM_NAMES` in
  `market_matcher.py`), rather than parsing the ticker structure, since
  ticker formats are more likely to change/vary than plain-English titles.
- Spread/total lines are read from `floor_strike` / `cap_strike` first,
  falling back to a regex over the title text.
- Home/away is inferred from an "X at Y" / "X vs Y" pattern in the title.

Before trusting the output, run `python -m kalshi_edge.main --dump-unmatched`
against live markets and check that nothing you expected to see got
skipped. If Kalshi's actual field format differs, the fix is localized to
`market_matcher.py` (`find_team`, `parse_matchup`, `_line_from_market`).

By contrast, `nfl_ratings.py` (via `nflreadpy`, which pulls from
nflverse's GitHub-hosted data) and all the math in `probability.py` /
`kelly.py` *were* run and validated end-to-end against live data from
this sandbox — those parts you can trust as-is.

## Before betting real money

- Backtest: run the model against closed markets / historical lines and
  check calibration before sizing real positions.
- Spread and total markets have historically traded thinner than the
  game-winner market on Kalshi — expect wider spreads and more slippage
  executing there.
- Kalshi's sports contracts sit in a legally contested spot (state-level
  access has been shifting) — confirm your state's current standing
  before funding an account.
- This is a starting point, not a finished trading system: the Elo/scoring
  constants in `config.py` are reasonable priors, not fitted/optimized
  parameters.
