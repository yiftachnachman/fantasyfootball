"""Walk-forward backtest against ACTUAL game outcomes, not just Vegas
closing lines (that's what backtest.py checks -- a calibration/bias
check against an efficient benchmark, not a measure of whether the
model would have made money).

This module answers: if you'd bet according to this model every week,
using only information available before each game, would you have won?
Uses nfl_ratings.build_ratings(..., record_predictions=True), which
captures each game's prediction from the model's state BEFORE that
game's result is folded in -- no leakage of the future into the past,
unlike naively scoring historical games with final, all-seasons ratings.

Since historical Kalshi prices for past seasons don't exist, spread/total
bets are evaluated against Vegas's closing line at standard -110 odds (a
well-known industry baseline), not simulated Kalshi pricing -- this
measures "does the model beat a sharp closing line," a meaningfully
harder bar than beating Kalshi's thinner markets.

Usage:
    python -m kalshi_edge.backtest_outcomes
    python -m kalshi_edge.backtest_outcomes --seasons 2010 2011 ... 2025 --min-edge 0.05
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from .config import SETTINGS
from .nfl_ratings import build_ratings
from .probability import cover_probability, over_probability

VIG_STAKE = 110.0  # standard American-odds "bet 110 to win 100"
VIG_WIN = 100.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=int, nargs="+", default=list(range(2010, 2026)),
                    help="seasons to build ratings from and backtest against")
    p.add_argument("--warmup-seasons", type=int, default=1,
                    help="drop this many earliest seasons from scoring (ratings are cold-start noise then)")
    p.add_argument("--min-edge", type=float, default=0.05,
                    help="only place a simulated ATS/O-U bet when the model's probability is at least this far from 50%%")
    return p.parse_args(argv)


def _brier_and_logloss(prob: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    brier = float(np.mean((prob - outcome) ** 2))
    eps = 1e-9
    clipped = np.clip(prob, eps, 1 - eps)
    logloss = float(-np.mean(outcome * np.log(clipped) + (1 - outcome) * np.log(1 - clipped)))
    return brier, logloss


def _calibration_table(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"prob": prob, "outcome": outcome})
    df["bin"] = pd.cut(df["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("outcome", "size"), predicted=("prob", "mean"), actual=("outcome", "mean"),
    )
    return grouped[grouped["n"] > 0]


def _simulate_line_bets(model_prob: pd.Series, outcome: pd.Series, push: pd.Series, min_edge: float) -> dict:
    """model_prob: P(side A) at the line. outcome: 1 if side A happened, 0 if side B, ignored where push."""
    valid = ~push
    prob, out = model_prob[valid], outcome[valid]
    bet_a = prob >= 0.5
    won = np.where(bet_a, out == 1, out == 0)

    def _summary(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n_bets": 0, "win_rate": None, "roi_pct": None}
        wins = int(won[mask].sum())
        profit = wins * VIG_WIN - (n - wins) * VIG_STAKE
        return {"n_bets": n, "win_rate": wins / n, "roi_pct": 100 * profit / (n * VIG_STAKE)}

    all_mask = np.ones(len(prob), dtype=bool)
    edge_mask = (prob - 0.5).abs().to_numpy() >= min_edge
    return {"all": _summary(all_mask), "edge_filtered": _summary(edge_mask)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Building walk-forward predictions for seasons {args.seasons}...")
    _, preds = build_ratings(seasons=args.seasons, record_predictions=True)

    keep_seasons = sorted(set(args.seasons))[args.warmup_seasons:]
    preds = preds[preds["season"].isin(keep_seasons)].copy()
    print(f"Scoring {len(preds)} games from {keep_seasons[0]}-{keep_seasons[-1]} "
          f"(dropped {args.warmup_seasons} warm-up season(s) for cold-start ratings noise).\n")

    # --- Moneyline: Brier score, log loss, calibration ---
    ml = preds.dropna(subset=["home_score", "away_score"]).copy()
    ml = ml[ml["home_score"] != ml["away_score"]]  # drop ties, undefined win/loss target
    outcome = (ml["home_score"] > ml["away_score"]).astype(float).to_numpy()
    prob = ml["home_win_prob"].to_numpy()
    brier, logloss = _brier_and_logloss(prob, outcome)
    print("=== Moneyline (home win probability) ===")
    print(f"n={len(ml)}  Brier score={brier:.4f} (lower is better, 0.25 = coin flip)  log loss={logloss:.4f}")
    print("\nCalibration (predicted vs actual home win rate by decile):")
    print(_calibration_table(prob, outcome).to_string(float_format="{:.3f}".format))

    # --- Spread: beat the Vegas closing line? ---
    spread = preds.dropna(subset=["spread_line", "home_score", "away_score"]).copy()
    spread["actual_margin"] = spread["home_score"] - spread["away_score"]
    spread["push"] = spread["actual_margin"] == spread["spread_line"]
    spread["model_home_cover_prob"] = spread.apply(
        lambda r: cover_probability(r["predicted_margin"], r["spread_line"]), axis=1,
    )
    spread["home_covered"] = (spread["actual_margin"] > spread["spread_line"]).astype(float)
    ats = _simulate_line_bets(spread["model_home_cover_prob"], spread["home_covered"], spread["push"], args.min_edge)
    print(f"\n=== Spread (vs. Vegas closing line, standard -110 odds) ===")
    print(f"All games:        n={ats['all']['n_bets']}  win rate={ats['all']['win_rate']:.1%}  ROI={ats['all']['roi_pct']:+.1f}%")
    if ats["edge_filtered"]["n_bets"]:
        print(f"Edge >= {args.min_edge:.0%}:   n={ats['edge_filtered']['n_bets']}  win rate={ats['edge_filtered']['win_rate']:.1%}  ROI={ats['edge_filtered']['roi_pct']:+.1f}%")
    else:
        print(f"Edge >= {args.min_edge:.0%}:   no games cleared this threshold")
    print("\nCalibration (model's home-cover probability vs actual home-cover rate, by decile):")
    print(_calibration_table(spread["model_home_cover_prob"].to_numpy(), spread["home_covered"].to_numpy()).to_string(float_format="{:.3f}".format))

    # --- Total: beat the Vegas closing line? ---
    total = preds.dropna(subset=["total_line", "home_score", "away_score"]).copy()
    total["actual_total"] = total["home_score"] + total["away_score"]
    total["push"] = total["actual_total"] == total["total_line"]
    total["model_over_prob"] = total.apply(
        lambda r: over_probability(r["predicted_total"], r["total_line"]), axis=1,
    )
    total["went_over"] = (total["actual_total"] > total["total_line"]).astype(float)
    ou = _simulate_line_bets(total["model_over_prob"], total["went_over"], total["push"], args.min_edge)
    print(f"\n=== Total (vs. Vegas closing line, standard -110 odds) ===")
    print(f"All games:        n={ou['all']['n_bets']}  win rate={ou['all']['win_rate']:.1%}  ROI={ou['all']['roi_pct']:+.1f}%")
    if ou["edge_filtered"]["n_bets"]:
        print(f"Edge >= {args.min_edge:.0%}:   n={ou['edge_filtered']['n_bets']}  win rate={ou['edge_filtered']['win_rate']:.1%}  ROI={ou['edge_filtered']['roi_pct']:+.1f}%")
    else:
        print(f"Edge >= {args.min_edge:.0%}:   no games cleared this threshold")
    print("\nCalibration (model's over probability vs actual over rate, by decile):")
    print(_calibration_table(total["model_over_prob"].to_numpy(), total["went_over"].to_numpy()).to_string(float_format="{:.3f}".format))

    print(
        "\nA positive ROI here means the model beat a sharp closing line over this "
        "sample -- a real (if noisy, given sample size) signal. Kalshi itself is "
        "thinner than Vegas, so real performance there could differ in either "
        "direction. Breakeven at -110 odds is 52.38% win rate."
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
