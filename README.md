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

## Model tightening (validated against 15 seasons of real outcomes)

`kalshi_edge/backtest_outcomes.py` is a proper walk-forward backtest: it
uses `build_ratings(..., record_predictions=True)`, which captures each
game's prediction from only the information available *before* that game
(current Elo/scoring state, prior to that game's own result being folded
in) -- unlike naively scoring historical games with final, all-seasons
ratings, which leaks the future into the past and overstates accuracy.
Run it yourself:

```bash
python -m kalshi_edge.backtest_outcomes
python -m kalshi_edge.backtest_outcomes --seasons 2015 2016 ... 2025 --min-edge 0.05
```

Using this tool against 2011-2025 (n≈4,000 games), two real, evidence-backed
changes were made:

1. **Elo now regresses 1/3 toward the mean at every season transition**
   (`config.elo_season_regression`, was 0 i.e. off). Without it, games the
   model called 10-30% likely actually happened ~37% of the time --
   badly overconfident, because a team's rating carries over between
   seasons with no discount for roster turnover. Testing regression
   fractions from 0 to 0.45 found 1/3 minimized Brier score (0.2301 ->
   0.2224) and fixed that miscalibration almost exactly (predicted 26.0%
   vs actual 25.3% in that same bucket) -- also matching 538's published
   NFL Elo methodology, independently confirmed here rather than just
   borrowed.
2. **`margin_sigma`/`total_sigma` are now fit to this model's own
   walk-forward residual std against actual outcomes** (13.5 and 13.7),
   not generic literature figures (13.86 and 10.5). `total_sigma` in
   particular was badly off: the simple off/def scoring-average blend
   this model uses for totals carries meaningfully more of its own
   estimation noise than a tight literature prior assumes, and using too
   small a sigma made every over/under probability overconfident.

One thing investigated but deliberately **not** changed: the model's
predicted_total ran ~0.45-1 pt higher than Vegas's total_line. Checking
against actual game totals (not Vegas) showed the model is
unbiased against reality (-0.002 pts mean error); it's Vegas's own total
line that runs low (-0.455 pts vs actual) -- a well-documented effect
where books shade totals down because the public leans "over." Chasing
that number would mean deliberately introducing a real bias just to
match a benchmark's own known skew, so predicted_total was left alone.

### What this backtest says about beating Kalshi/Vegas

Read the output carefully -- it separates two different questions that
are easy to conflate:

- **Is the model's own probability an accurate description of reality?**
  Yes, reasonably -- the moneyline calibration table (predicted vs. actual
  win rate by decile) tracks closely, and Brier score (0.222) beats a
  coin flip (0.25).
- **Does disagreeing with Vegas's closing line predict beating it?** No.
  The spread/total ATS win rates hover at 49-51% (breakeven at standard
  -110 odds is 52.38%) and, tellingly, the calibration curves against
  Vegas are close to *flat* -- games the model was 70-80% confident in
  beat the Vegas line at about the same rate as games it was 40-50%
  confident in. That's the expected signature of "no extractable edge
  against a sharp closing line," not a bug: Vegas already incorporates
  most of what this model knows (and much it doesn't -- real injury
  reports, coaching intel, line movement from sharp money). A
  well-calibrated model and a market-beating model are not the same
  thing, and this one is (evidence says) the former, not the latter,
  against Vegas specifically. Real performance against Kalshi's thinner,
  less professionally-priced markets could differ -- that's an empirical
  question this backtest can't answer, since historical Kalshi prices
  for past seasons don't exist to test against.

## Checking calibration

`python -m kalshi_edge.backtest` compares the model's predicted
margin/total against Vegas's actual closing lines (`spread_line` /
`total_line`, both included in nflreadpy's schedule data) across whatever
seasons you point it at. Vegas closing lines are about as efficient a
benchmark as exists, so a systematic (not just noisy) disagreement there
is a real calibration bug worth fixing — as opposed to disagreeing with
Kalshi itself, which just as easily means the *model* is wrong, not the
market.

As of this writing: predicted margin has ~0 bias against Vegas (std ~5.5
pts — the model and Vegas simply disagree by that much on a typical
game, since this model has no injury/weather/beat-reporter information).
Predicted total runs a bit high against Vegas's line specifically, but
that's Vegas's line being shaded, not a model bug -- see "Model
tightening" below for the investigation and `backtest_outcomes.py` for a
proper accuracy/ROI backtest against real outcomes (this script only
checks bias against Vegas, a narrower and different question).
`config.elo_home_field` was fit using this script; rerun it before trusting
`min-edge` output if you change `nfl_ratings.py` or roll to new seasons.

## How market parsing actually works (verified against a live pull)

The sandbox that wrote the first version of this tool couldn't reach
Kalshi's API, so the original `market_matcher.py` was a best guess and
matched 0 of 724 live contracts on first real use. It's since been
rewritten against real API responses and verified end-to-end. Notable
things that differ from the naive assumption:

- **Prices** come back as decimal-dollar strings (`yes_ask_dollars =
  "0.6300"`), not a plain cents integer.
- **Title text names only one team** ("Kansas City wins", "Detroit wins
  by over 7.5 points?") — there's no "X at Y" pattern to parse, and
  Kalshi's market data never states which team is home.
- **Team identity comes from ticker structure, not text.** The final
  ticker segment names the contract's team directly (`...-KC`,
  `...-DET8`), and the event ticker's matchup code (`26SEP20INDKC` = date
  + concatenated team codes) lets you recover the opponent by
  subtracting the known team's code from either end — no fuzzy string
  matching needed. `rules_primary` text ("NY Giants vs LA Rams") is kept
  only as a fallback for total markets, which have no per-team ticker
  suffix.
- **Home/away is resolved against the real NFL schedule** (via
  `nflreadpy`), not guessed from Kalshi text: `TeamRatings.matchups` maps
  each team pair to every scheduled meeting (played or not — future weeks
  already have home/away assigned), and `market_matcher` looks up the one
  closest to the ticker's date.

If Kalshi changes their schema again, `python -m kalshi_edge.main
--dump-unmatched` prints a specific reason per skipped row (e.g. "no
schedule match for KC vs IND"), which is exactly what surfaced these
issues the first time.

## Situational adjustments (QB status, injuries, weather)

`kalshi_edge/situational.py` layers three signals on top of the base
Elo/scoring model, each as an Elo-point penalty (QB/other injuries) or a
predicted-total shift (weather):

- **QB status**: detected from `nflreadpy.load_depth_charts()`, not the
  weekly injury report (which lags -- verified live: on 2026-09-13 it only
  covered week 1 while week 2 markets were already open). A team's
  *current* depth-chart QB1 is compared against who was QB1 at the start
  of the regular season (~Sept 1) to catch an in-season change. That
  cutoff matters: comparing against the full offseason instead flagged 4
  ordinary free-agency signings as if they were injury news, confirmed
  live before the fix went in. The weekly injury report still contributes
  a smaller "Questionable" tier and an "Out/Doubtful" tier, but *only* for
  the specific week each market is about (`week_for_date`, matched
  against the schedule) -- applying one week's report to a different
  week's game is a real bug (also caught live: week 1 had Miami's
  starting QB listed Out, which must not bleed into week 2+ markets).
- **Other injuries**: a coarse count of non-QB "Out" starters, same
  week-scoping, capped at a small total Elo effect (`config.py`) -- this
  is a low-confidence signal by design; it can't tell a franchise left
  tackle from a backup long-snapper being out.
- **Weather**: outdoor-stadium games (see `situational.DOME_TEAMS`) get a
  live forecast from Open-Meteo (free, no API key), reducing predicted
  total for high wind/precipitation/cold. Domes and retractable-roof
  stadiums are excluded entirely rather than guessed at.

Enabled by default; pass `--no-situational` to `main.py` to score with
the base model only. Every signal fails open (a lookup error contributes
zero adjustment, never a crash). All three signals -- QB status, injuries,
and the Open-Meteo weather call -- have been verified against live data:
the sandbox that wrote this code couldn't reach Open-Meteo itself, but a
run on a machine with real internet confirmed it returns real forecasts
and correctly reduces predicted totals for rain/wind/cold.

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
