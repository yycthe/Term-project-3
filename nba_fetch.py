"""
nba_fetch.py — Live NBA data layer
===================================
Provides three capabilities:
  1. fetch_latest_games()  — pull completed games from NBA API, append to CSV
  2. get_team_next_game()  — find a team's next unplayed game from the schedule
  3. settle_predictions()  — check pending predictions against actual results
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime

import config

logger = logging.getLogger(__name__)

# ── NBA API imports ──────────────────────────────────────────────────────────
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.static import teams as nba_teams_static

# ── Constants ────────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_all_teams():
    """
    Return the official list of 30 NBA teams.
    Each entry: {id, full_name, abbreviation, nickname, city, state, year_founded}
    """
    return nba_teams_static.get_teams()


def _team_abbr_to_id():
    """Build a mapping abbr -> team_id (str)."""
    return {t["abbreviation"]: str(t["id"]) for t in get_all_teams()}


def _team_id_to_abbr():
    """Build a mapping team_id (str) -> abbr."""
    return {str(t["id"]): t["abbreviation"] for t in get_all_teams()}


# ═════════════════════════════════════════════════════════════════════════════
#  1. FETCH LATEST GAMES  (for agent.py — data refresh)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_latest_games(csv_path=None):
    """
    Pull completed NBA games from the stats API that are newer than the
    latest date already in the CSV.  Appends new rows and returns the count.

    The rolling-stat columns are left as NaN — features.py recomputes them
    during training, so there is no duplication.
    """
    if csv_path is None:
        csv_path = config.DATA_PATH

    # ── Read existing data ────────────────────────────────────────────────
    try:
        existing = pd.read_csv(csv_path)
        existing["GAME_DATE"] = pd.to_datetime(existing["GAME_DATE"])
        latest_date = existing["GAME_DATE"].max()
        existing_game_ids = set(existing["GAME_ID"].astype(str).unique())
    except Exception:
        logger.warning("Could not read existing CSV — will create from scratch.")
        existing = pd.DataFrame()
        latest_date = pd.Timestamp("2014-01-01")
        existing_game_ids = set()

    logger.info(f"Latest date in CSV: {latest_date.date()}")

    # ── Fetch current season from NBA API ─────────────────────────────────
    season = config.NBA_CURRENT_SEASON
    try:
        time.sleep(0.7)  # rate-limit courtesy
        log = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
        )
        api_df = log.get_data_frames()[0]
    except Exception as e:
        logger.error(f"NBA API fetch failed: {e}")
        return 0, None

    if api_df.empty:
        logger.info("API returned no games.")
        return 0, None

    api_df["GAME_DATE"] = pd.to_datetime(api_df["GAME_DATE"])

    # Filter: only games newer than what we have AND not already present
    new_games = api_df[
        (api_df["GAME_DATE"] > latest_date)
        & (~api_df["GAME_ID"].astype(str).isin(existing_game_ids))
    ].copy()

    if new_games.empty:
        latest_date_after = existing["GAME_DATE"].max() if not existing.empty else latest_date
        logger.info("No new games to add.")
        return 0, latest_date_after

    # ── Add derived columns to match existing CSV schema ──────────────────
    season_year = int(season.split("-")[0])
    new_games["SEASON_START_YEAR"] = season_year
    new_games["SEASON_TYPE"] = "Regular Season"
    new_games["WIN"] = (new_games["WL"] == "W").astype(int)
    new_games["IS_HOME"] = new_games["MATCHUP"].apply(
        lambda x: "vs." in str(x)
    )

    # Align columns — keep only columns that exist in the CSV
    if not existing.empty:
        for col in existing.columns:
            if col not in new_games.columns:
                new_games[col] = np.nan
        # Keep in the same column order, drop extras from API
        new_games = new_games[[c for c in existing.columns if c in new_games.columns]]

    # ── Append & save ─────────────────────────────────────────────────────
    combined = pd.concat([existing, new_games], ignore_index=True)
    combined.to_csv(csv_path, index=False)

    n_new = len(new_games)
    latest_date_after = combined["GAME_DATE"].max() if not combined.empty else None
    logger.info(f"Appended {n_new} new game rows to {csv_path}; data through {latest_date_after.date() if latest_date_after is not None else 'N/A'}")
    return n_new, latest_date_after


# ═════════════════════════════════════════════════════════════════════════════
#  2. SCHEDULE — upcoming games
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_full_schedule():
    """Fetch the full NBA season schedule from the CDN (cached for 5 min)."""
    try:
        resp = requests.get(SCHEDULE_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Schedule fetch failed: {e}")
        return None


def _parse_schedule_date(raw):
    """Parse the schedule date string which can have several formats."""
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    # Last resort: try pandas
    try:
        return pd.to_datetime(raw).to_pydatetime()
    except Exception:
        return None


def get_team_next_game(team_abbr):
    """
    Find the next unplayed game for *team_abbr* (e.g. 'LAL').

    Returns a dict:
        {game_id, game_date, home_team, away_team,
         home_team_id, away_team_id, game_status_text}
    or None if nothing is found (off-season, schedule unavailable).
    """
    schedule = _fetch_full_schedule()
    if not schedule:
        return None

    today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    game_dates = schedule.get("leagueSchedule", {}).get("gameDates", [])

    for gd in game_dates:
        game_dt = _parse_schedule_date(gd.get("gameDate", ""))
        if game_dt is None:
            continue
        # Include today's games and future games (>= today)
        if game_dt.date() < today_dt.date():
            continue

        for game in gd.get("games", []):
            # gameStatus: 1 = scheduled, 2 = in progress, 3 = final
            # Skip only final games; include scheduled and in-progress
            if game.get("gameStatus", 0) == 3:
                continue

            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            h_abbr = home.get("teamTricode") or ""
            a_abbr = away.get("teamTricode") or ""

            # Skip if team abbreviations are missing
            if not h_abbr or not a_abbr:
                continue

            if team_abbr.upper() in (h_abbr.upper(), a_abbr.upper()):
                return {
                    "game_id": game.get("gameId", ""),
                    "game_date": game_dt.strftime("%Y-%m-%d"),
                    "home_team": h_abbr,
                    "away_team": a_abbr,
                    "home_team_id": str(home.get("teamId", "")),
                    "away_team_id": str(away.get("teamId", "")),
                    "game_status_text": game.get("gameStatusText", ""),
                }

    return None


def get_team_upcoming_games(team_abbr, limit=5):
    """Return the next *limit* unplayed games for a team."""
    schedule = _fetch_full_schedule()
    if not schedule:
        return []

    today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    game_dates = schedule.get("leagueSchedule", {}).get("gameDates", [])
    results = []

    for gd in game_dates:
        game_dt = _parse_schedule_date(gd.get("gameDate", ""))
        if game_dt is None or game_dt < today_dt:
            continue

        for game in gd.get("games", []):
            if game.get("gameStatus", 0) == 3:
                continue
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            h_abbr = home.get("teamTricode", "")
            a_abbr = away.get("teamTricode", "")

            if team_abbr.upper() in (h_abbr.upper(), a_abbr.upper()):
                results.append({
                    "game_id": game.get("gameId", ""),
                    "game_date": game_dt.strftime("%Y-%m-%d"),
                    "home_team": h_abbr,
                    "away_team": a_abbr,
                    "home_team_id": str(home.get("teamId", "")),
                    "away_team_id": str(away.get("teamId", "")),
                    "game_status_text": game.get("gameStatusText", ""),
                })
                if len(results) >= limit:
                    return results

    return results


def get_all_upcoming_games(limit_days=7):
    """
    Return all unplayed games in the next limit_days days (league-wide).
    Each game: {game_id, game_date, home_team, away_team, home_team_id, away_team_id, game_status_text}.
    """
    from datetime import timedelta
    schedule = _fetch_full_schedule()
    if not schedule:
        return []
    today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today_dt + timedelta(days=limit_days)
    game_dates = schedule.get("leagueSchedule", {}).get("gameDates", [])
    results = []
    for gd in game_dates:
        game_dt = _parse_schedule_date(gd.get("gameDate", ""))
        if game_dt is None or game_dt.date() < today_dt.date():
            continue
        if game_dt.date() > cutoff.date():
            break
        for game in gd.get("games", []):
            if game.get("gameStatus", 0) == 3:
                continue
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            h_abbr = home.get("teamTricode") or ""
            a_abbr = away.get("teamTricode") or ""
            if not h_abbr or not a_abbr:
                continue
            results.append({
                "game_id": game.get("gameId", ""),
                "game_date": game_dt.strftime("%Y-%m-%d"),
                "home_team": h_abbr,
                "away_team": a_abbr,
                "home_team_id": str(home.get("teamId", "")),
                "away_team_id": str(away.get("teamId", "")),
                "game_status_text": game.get("gameStatusText", ""),
            })
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  3. SETTLE PREDICTIONS  (check pending predictions against results)
# ═════════════════════════════════════════════════════════════════════════════

def settle_predictions(predictions):
    """
    For every prediction with status == 'pending', check if the game has
    been played.  If so, fill in actual_winner / correct / status.

    Uses the NBA CDN schedule which includes scores for completed games.
    Returns the (potentially modified) predictions list.
    """
    pending = [p for p in predictions if p.get("status") == "pending"]
    if not pending:
        return predictions

    schedule = _fetch_full_schedule()
    if not schedule:
        return predictions

    # Build lookup: game_id -> game data
    game_lookup = {}
    for gd in schedule.get("leagueSchedule", {}).get("gameDates", []):
        for game in gd.get("games", []):
            gid = game.get("gameId", "")
            if gid:
                game_lookup[gid] = game

    for pred in predictions:
        if pred.get("status") != "pending":
            continue

        game = game_lookup.get(pred.get("game_id", ""))
        if not game:
            continue

        if game.get("gameStatus", 0) != 3:
            continue

        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        def _safe_int(v):
            try:
                return int(v)
            except Exception:
                return None

        h_score = _safe_int(home.get("score"))
        a_score = _safe_int(away.get("score"))
        if h_score is None or a_score is None:
            continue

        if h_score > a_score:
            actual_winner = home.get("teamTricode", pred.get("home_team"))
        else:
            actual_winner = away.get("teamTricode", pred.get("away_team"))

        pred["actual_winner"] = actual_winner
        pred["correct"] = pred.get("predicted_winner", "") == actual_winner
        pred["status"] = "settled"
        pred["home_score"] = h_score
        pred["away_score"] = a_score

    return predictions


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTIONS I/O
# ═════════════════════════════════════════════════════════════════════════════

def load_predictions():
    """Load predictions from the JSON file."""
    path = config.PREDICTIONS_FILE
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_predictions(predictions):
    """Save predictions to the JSON file."""
    path = config.PREDICTIONS_FILE
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
