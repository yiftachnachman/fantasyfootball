"""Entry point: pull Kalshi NFL markets, score them against the model,
and print/save a ranked list of edges.

Usage:
    python -m kalshi_edge.main
    python -m kalshi_edge.main --min-edge 0.05 --bankroll 500
    python -m kalshi_edge.main --dump-unmatched   # inspect rows the parser skipped
    python -m kalshi_edge.main --no-situational   # base model only, skip QB/injury/weather adjustments
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import SETTINGS
from .kalshi_client import pull_all_nfl_markets
from .kelly import kelly_stake
from .market_matcher import evaluate_markets
from .nfl_ratings import build_ratings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-edge", type=float, default=SETTINGS.min_edge)
    p.add_argument("--min-volume", type=int, default=SETTINGS.min_volume)
    p.add_argument("--bankroll", type=float, default=SETTINGS.bankroll)
    p.add_argument("--kelly-multiplier", type=float, default=SETTINGS.kelly_multiplier)
    p.add_argument("--seasons", type=int, nargs="+", default=None,
                    help="explicit list of seasons for the ratings model, e.g. --seasons 2022 2023 2024")
    p.add_argument("--out", default="nfl_edges.csv", help="CSV path for the full ranked output")
    p.add_argument("--dump-unmatched", action="store_true",
                    help="print raw fields for markets the parser couldn't match, then exit")
    p.add_argument("--no-situational", action="store_true",
                    help="skip QB-status/injury/weather adjustments, score with the base model only")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Pulling open NFL markets from Kalshi...")
    markets = pull_all_nfl_markets()
    print(f"  {len(markets)} open contracts across {markets['bet_type'].nunique()} bet types")

    print(f"Building team ratings from the last {SETTINGS.seasons_of_history} seasons of nflreadpy history...")
    ratings = build_ratings(seasons=args.seasons)

    if args.dump_unmatched:
        from .market_matcher import unmatched_reason

        unmatched_count = 0
        for m in markets.to_dict("records"):
            reason = unmatched_reason(m, ratings)
            if reason:
                unmatched_count += 1
                print(f"[unmatched: {reason}] {m['bet_type']:9s} ticker={m['ticker']!r} "
                      f"title={m['title']!r} rules_primary={m.get('rules_primary')!r}")
        print(f"\n{unmatched_count} of {len(markets)} markets unmatched.")
        return 0

    context = None
    if not args.no_situational:
        print("Building situational context (QB status, injuries, weather)...")
        from .situational import build_context

        # Always the current season only -- depth-chart/injury history from
        # prior years isn't relevant to "who is the QB right now", and
        # mixing seasons would break the baseline-vs-current comparison.
        context = build_context(seasons=[pd.Timestamp.now("UTC").year])
        changed = [
            f"{team} ({context.current_qb1_name.get(team, '?')})"
            for team in context.current_qb1
            if context.qb_penalty_elo(team) > 0
        ]
        if changed:
            print(f"  QB-situation adjustment applied for: {', '.join(sorted(changed))}")
        else:
            print("  No QB-situation adjustments detected.")

    print("Scoring markets against the model...")
    edges = evaluate_markets(markets, ratings, context=context)
    if edges.empty:
        print("No markets could be matched to the model. Run with --dump-unmatched "
              "to see the raw fields and adjust kalshi_edge/market_matcher.py.")
        return 1

    edges = edges[edges["volume"] >= args.min_volume]
    edges["stake"] = edges.apply(
        lambda r: kelly_stake(
            r["model_prob"], r["yes_ask"],
            bankroll=args.bankroll, kelly_multiplier=args.kelly_multiplier,
        ),
        axis=1,
    )
    edges = edges.sort_values("edge", ascending=False)
    edges.to_csv(args.out, index=False)

    playable = edges[edges["edge"].abs() >= args.min_edge]
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nFull ranked list saved to {args.out} ({len(edges)} rows).")
    print(f"\n{len(playable)} contracts clear the {args.min_edge:.0%} edge threshold:\n")
    if len(playable):
        print(playable[["bet_type", "side", "model_prob", "market_prob", "edge", "yes_ask", "stake"]]
              .to_string(index=False, formatters={
                  "model_prob": "{:.1%}".format,
                  "market_prob": "{:.1%}".format,
                  "edge": "{:+.1%}".format,
                  "stake": "${:.2f}".format,
              }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
