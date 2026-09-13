"""Check model calibration against Vegas closing lines.

nflreadpy's schedule data already includes the closing spread_line and
total_line for every completed game, which makes for a much better sanity
check than staring at Kalshi edges: Vegas lines are about as efficient a
benchmark as exists, so if the model systematically diverges from them in
one direction, that's a real bug/miscalibration to fix -- as opposed to
disagreeing with them by some spread with no particular direction, which
just reflects the model having less information (no injuries, weather,
etc.) than a real sportsbook line.

IMPORTANT sign convention (verified against live nflreadpy data, since
this is exactly the kind of thing easy to get backwards): spread_line is
the home team's expected margin directly -- positive means the home team
is favored by that many points. This is the opposite of how a sportsbook
board displays a spread (where the favorite gets a "-" number); trust
this module's convention over intuition here.

This uses each game's FINAL ratings (built from all loaded seasons) to
predict earlier games in the same window, which is optimistic/leaky --
it's a bias check, not a true walk-forward accuracy backtest. A bias near
zero is meaningful either way; don't read the accuracy/std numbers as a
promise of real-world performance.

Usage:
    python -m kalshi_edge.backtest
    python -m kalshi_edge.backtest --seasons 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse

import numpy as np

from .nfl_ratings import build_ratings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025],
                    help="seasons to build ratings from AND backtest against")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import nflreadpy as nfl

    print(f"Building ratings from seasons {args.seasons}...")
    ratings = build_ratings(seasons=args.seasons)

    schedules = nfl.load_schedules(seasons=args.seasons)
    if hasattr(schedules, "to_pandas"):
        schedules = schedules.to_pandas()
    played = schedules.dropna(subset=["home_score", "away_score", "spread_line", "total_line"])

    margin_errs, total_errs = [], []
    for row in played.itertuples():
        pred = ratings.predict_game(row.home_team, row.away_team)
        vegas_margin = row.spread_line  # home-team-favored-by convention, see module docstring
        margin_errs.append(pred["predicted_margin"] - vegas_margin)
        total_errs.append(pred["predicted_total"] - row.total_line)

    margin_errs = np.array(margin_errs)
    total_errs = np.array(total_errs)
    n = len(margin_errs)
    print(f"\n{n} completed games with both a model prediction and a Vegas closing line.\n")
    print(f"Predicted margin vs. Vegas spread_line:")
    print(f"  mean bias = {margin_errs.mean():+.2f} pts  (std err {margin_errs.std()/n**0.5:.2f})")
    print(f"  std       = {margin_errs.std():.2f} pts")
    print(f"\nPredicted total vs. Vegas total_line:")
    print(f"  mean bias = {total_errs.mean():+.2f} pts  (std err {total_errs.std()/n**0.5:.2f})")
    print(f"  std       = {total_errs.std():.2f} pts")
    print(
        "\nA |mean bias| more than ~2 standard errors from zero suggests a real "
        "calibration issue worth fixing (e.g. elo_home_field, elo_points_per_elo "
        "in config.py). The std values reflect the model's disagreement with "
        "Vegas, not its accuracy against actual outcomes -- some disagreement "
        "is expected since this model has far less information than a real book."
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
