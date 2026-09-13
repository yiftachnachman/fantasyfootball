"""Injury/QB-status and weather adjustments layered on top of the base
Elo/scoring model in nfl_ratings.py.

Two independent signals, each converted to an Elo-point penalty (QB/other
injuries) or a predicted-total delta (weather), then folded into a
prediction via adjust_prediction():

- QB status: detected from nflreadpy.load_depth_charts(), not the weekly
  injury report. The injury report is filed a few days before each game,
  so on the day markets open for the upcoming week it often doesn't cover
  that week yet (verified live: on 2026-09-13, load_injuries only had
  data through week 1 while week 2 markets were already open). The depth
  chart updates roughly daily year-round, so comparing a team's CURRENT
  QB1 against who was QB1 at the start of the regular season catches an
  in-season starter change (injury or benching) well before the injury
  report would. The baseline is deliberately anchored at kickoff (roughly
  Sept 1), not the earliest depth-chart snapshot available (which reaches
  back into March): comparing against the offseason would flag ordinary
  free-agency signings and trades -- already public, already priced in
  -- as if they were fresh injury news. Verified live: without this
  cutoff, four teams falsely showed a "QB change" that were just normal
  offseason moves (e.g. a free agent QB signing); with it, zero false
  positives on the same data. The injury report is still used as a
  secondary signal for "Questionable" (smaller penalty) and to catch an
  Out/Doubtful starter who hasn't technically been replaced in the depth
  chart yet -- but ONLY for the week that report actually covers. Injury
  reports are per-week; naively using "whatever the latest available
  week is" for every market regardless of which week it's about is a
  real bug, not a simplification -- verified live: on 2026-09-13, the
  week-1 report had Miami's Tua Tagovailoa listed Out (he'd already
  played that game), and blindly applying that to a week-2+ market would
  wrongly penalize a team over a status that may no longer hold. So the
  injury-report tier is looked up by the specific week each game falls
  in (via week_for_date(), matched against the schedule), and simply
  contributes nothing when that week isn't covered yet.
- Weather: outdoor-stadium games (see DOME_TEAMS for the exclusion list)
  get a live forecast from Open-Meteo (free, no API key) for the game
  date, converted into a reduction of predicted_total. Retractable-roof
  and dome stadiums are excluded entirely rather than guessed at, since
  game-day roof state isn't reliably knowable in advance.

Everything here fails open: a lookup that errors (network blocked,
unexpected data shape, unmapped team) contributes zero adjustment rather
than raising. The nflreadpy-based QB logic was run against live 2026 data
while writing this. The Open-Meteo call was NOT reachable from this
sandbox (same network restriction that blocks Kalshi -- see README) and
needs verification on a machine with real internet access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import SETTINGS
from .nfl_ratings import normalize_team

# Retractable-roof and fixed-dome stadiums, where we can't reliably know
# game-day roof state in advance -- treated as "no weather effect" rather
# than risk applying an outdoor penalty to what turns out to be a closed
# roof. LA and LAC share SoFi Stadium (fixed canopy, effectively enclosed).
DOME_TEAMS = frozenset({"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LAC", "LA", "MIN", "NO"})

# Approximate (lat, lon) of each outdoor team's home stadium -- city-level
# precision is plenty for a wind/temp/precip forecast.
TEAM_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DEN": (39.7439, -105.0201),
    "GB": (44.5013, -88.0622),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "MIA": (25.9580, -80.2389),
    "NE": (42.0909, -71.2643),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4033, -121.9694),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9077, -76.8645),
}

_STARTER_OUT_STATUSES = frozenset({"Out", "Doubtful", "IR"})


@dataclass
class SituationalContext:
    baseline_qb1: dict[str, str] = field(default_factory=dict)  # team -> gsis_id, as of start of season
    current_qb1: dict[str, str] = field(default_factory=dict)   # team -> gsis_id, as of latest depth chart
    current_qb1_name: dict[str, str] = field(default_factory=dict)  # team -> player name, for display only
    injury_status_by_week: dict[int, dict[tuple[str, str], str]] = field(default_factory=dict)
    _week_dates: list[tuple[pd.Timestamp, int]] = field(default_factory=list, repr=False)
    _weather_cache: dict[tuple[float, float, str], dict | None] = field(default_factory=dict, repr=False)

    def week_for_date(self, event_date: pd.Timestamp | None) -> int | None:
        """The schedule week whose games are closest to event_date, or
        None if there's no schedule data or nothing within 4 days (too
        far off to trust as "the same week")."""
        if event_date is None or not self._week_dates:
            return None
        gameday, week = min(self._week_dates, key=lambda gw: abs((gw[0] - event_date).days))
        return week if abs((gameday - event_date).days) <= 4 else None

    def qb_penalty_elo(self, team: str, week: int | None = None) -> float:
        """Non-negative Elo penalty for this team's QB situation. `week`
        scopes the injury-report tier (Questionable/Out) to a specific
        week -- omit it to only check the depth-chart change signal,
        which is not week-specific."""
        team = normalize_team(team)
        base = self.baseline_qb1.get(team)
        cur = self.current_qb1.get(team)
        if base and cur and base != cur:
            return SETTINGS.qb_change_elo_penalty
        if cur and week is not None:
            status = self.injury_status_by_week.get(week, {}).get((team, cur))
            if status == "Questionable":
                return SETTINGS.qb_questionable_elo_penalty
            if status in _STARTER_OUT_STATUSES:
                return SETTINGS.qb_change_elo_penalty
        return 0.0

    def key_injury_penalty_elo(self, team: str, week: int | None = None) -> float:
        """Non-negative Elo penalty from non-QB "Out" starters in the
        given week's injury report. Deliberately coarse (see module
        docstring); excludes the current QB1 so this doesn't double-count
        with qb_penalty_elo. 0.0 if week is None or not covered yet."""
        if week is None:
            return 0.0
        team = normalize_team(team)
        qb_gsis = self.current_qb1.get(team)
        week_statuses = self.injury_status_by_week.get(week, {})
        count = sum(
            1 for (t, gid), status in week_statuses.items()
            if t == team and status == "Out" and gid != qb_gsis
        )
        return min(count * SETTINGS.key_injury_elo_penalty_each, SETTINGS.key_injury_elo_penalty_cap)

    def weather_total_delta(self, home_team: str, event_date: pd.Timestamp | None) -> float:
        """Non-positive adjustment to predicted_total for the home team's
        stadium/date. 0.0 if the stadium is a dome, unmapped, or the
        forecast can't be fetched (too far out, network error, ...)."""
        home = normalize_team(home_team)
        if home in DOME_TEAMS or event_date is None:
            return 0.0
        coords = TEAM_STADIUM_COORDS.get(home)
        if not coords:
            return 0.0
        forecast = self._fetch_weather(*coords, event_date.strftime("%Y-%m-%d"))
        if not forecast:
            return 0.0
        delta = 0.0
        wind = forecast.get("windspeed_max")
        if wind is not None and wind > SETTINGS.weather_wind_threshold_mph:
            delta -= (wind - SETTINGS.weather_wind_threshold_mph) * SETTINGS.weather_wind_coeff
        precip = forecast.get("precipitation_sum")
        if precip is not None and precip > SETTINGS.weather_precip_threshold_mm:
            delta -= SETTINGS.weather_precip_total_penalty
        temp_min = forecast.get("temp_min")
        if temp_min is not None and temp_min < SETTINGS.weather_cold_threshold_f:
            delta -= (SETTINGS.weather_cold_threshold_f - temp_min) * SETTINGS.weather_cold_coeff
        return max(delta, -SETTINGS.weather_total_delta_cap)

    def _fetch_weather(self, lat: float, lon: float, date: str) -> dict | None:
        key = (lat, lon, date)
        if key in self._weather_cache:
            return self._weather_cache[key]
        result = None
        try:
            import requests

            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date,
                    "end_date": date,
                    "daily": "temperature_2m_min,windspeed_10m_max,precipitation_sum",
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                    "precipitation_unit": "mm",
                    "timezone": "auto",
                },
                timeout=10,
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            temps = daily.get("temperature_2m_min") or []
            winds = daily.get("windspeed_10m_max") or []
            precips = daily.get("precipitation_sum") or []
            if temps or winds or precips:
                result = {
                    "temp_min": temps[0] if temps else None,
                    "windspeed_max": winds[0] if winds else None,
                    "precipitation_sum": precips[0] if precips else None,
                }
        except Exception:
            result = None
        self._weather_cache[key] = result
        return result


def build_context(seasons: list[int]) -> SituationalContext:
    """Best-effort: any individual data source failing just leaves that
    signal empty rather than aborting the whole run."""
    import nflreadpy as nfl

    ctx = SituationalContext()

    try:
        dc = nfl.load_depth_charts(seasons=seasons)
        if hasattr(dc, "to_pandas"):
            dc = dc.to_pandas()
        qbs = dc[dc["pos_abb"] == "QB"].copy()
        qbs["dt"] = pd.to_datetime(qbs["dt"], utc=True)
        qbs["team"] = qbs["team"].map(normalize_team)
        # Anchor the baseline at kickoff, not the earliest snapshot
        # available (offseason) -- see module docstring.
        season_start = pd.Timestamp(year=min(seasons), month=9, day=1, tz="UTC")
        qbs = qbs[qbs["dt"] >= season_start]
        if len(qbs):
            latest_mask = qbs.groupby("team")["dt"].transform("max") == qbs["dt"]
            baseline_mask = qbs.groupby("team")["dt"].transform("min") == qbs["dt"]
            current_qb1 = qbs[latest_mask & (qbs["pos_rank"] == 1)]
            baseline_qb1 = qbs[baseline_mask & (qbs["pos_rank"] == 1)]
            ctx.current_qb1 = dict(zip(current_qb1["team"], current_qb1["gsis_id"]))
            ctx.current_qb1_name = dict(zip(current_qb1["team"], current_qb1["player_name"]))
            ctx.baseline_qb1 = dict(zip(baseline_qb1["team"], baseline_qb1["gsis_id"]))
    except Exception:
        pass

    try:
        sched = nfl.load_schedules(seasons=seasons)
        if hasattr(sched, "to_pandas"):
            sched = sched.to_pandas()
        sched = sched.dropna(subset=["gameday", "week"])
        ctx._week_dates = list(zip(pd.to_datetime(sched["gameday"]), sched["week"].astype(int)))
    except Exception:
        pass

    try:
        inj = nfl.load_injuries(seasons=seasons)
        if hasattr(inj, "to_pandas"):
            inj = inj.to_pandas()
        if len(inj):
            by_week: dict[int, dict[tuple[str, str], str]] = {}
            for week, t, gid, status in zip(inj["week"], inj["team"], inj["gsis_id"], inj["report_status"]):
                if pd.isna(status):
                    continue
                by_week.setdefault(int(week), {})[(normalize_team(t), gid)] = status
            ctx.injury_status_by_week = by_week
    except Exception:
        pass

    return ctx


def adjust_prediction(
    pred: dict,
    home_penalty_elo: float,
    away_penalty_elo: float,
    total_delta: float,
    points_per_elo: float = SETTINGS.elo_points_per_elo,
) -> dict:
    """Recompute win_prob/margin after applying non-negative Elo penalties
    (each representing how much weaker that team is expected to play),
    and shift predicted_total by total_delta (already signed, <= 0)."""
    elo_diff = pred["predicted_margin"] * points_per_elo
    elo_diff_adj = elo_diff - home_penalty_elo + away_penalty_elo
    home_win_prob = 1.0 / (1.0 + 10 ** (-elo_diff_adj / 400.0))
    return {
        "home_win_prob": home_win_prob,
        "away_win_prob": 1.0 - home_win_prob,
        "predicted_margin": elo_diff_adj / points_per_elo,
        "predicted_total": pred["predicted_total"] + total_delta,
    }
