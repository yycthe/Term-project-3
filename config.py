import os
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────
DATA_PATH = "nba_game_logs_combined.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

# ── Column Detection ───────────────────────────────────────
TARGET_CANDIDATES = ["win", "won", "result", "home_win", "y", "WIN"]
DATE_CANDIDATES = ["date", "game_date", "datetime", "match_date", "GAME_DATE"]
TEAM_CANDIDATES = ["home_team", "away_team", "team", "opponent", "TEAM_ID", "TEAM_ABBREVIATION"]
SEASON_COL = "SEASON_START_YEAR"

# ── Leakage Patterns ──────────────────────────────────────
LEAKAGE_KEYWORDS = [
    "score", "pts", "points", "goals", "margin", "wl", "w_pct", "w", "l",
    "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
    "ftm", "fta", "ft_pct", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"
]

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Core Settings ──────────────────────────────────────────
RANDOM_STATE = 42

# Optimisation target: minimise LogLoss (probability quality)
# with an AUC floor to ensure ranking ability is not sacrificed.
LOGLOSS_THRESHOLD = 0.63   # Goal: LogLoss <= this → stop
AUC_FLOOR = 0.60           # Constraint: trials with AUC < this are penalised
AUC_THRESHOLD = 0.68       # Reference only (legacy / monitoring)

# ── Hyperparameter Search Space (defaults — Phase 1) ──────
SEARCH_WINDOW_YEARS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEARCH_ELO_K = [5, 10, 15, 20, 25, 30, 35, 40, 50]
# Rolling stats: number of games to average (e.g. 10 = last 10 games)
SEARCH_ROLL_WINDOW = [5, 10, 15, 20]
ROLL_WINDOW_DEFAULT = 10
SEARCH_MODELS = ["XGBoost", "RandomForest", "LogisticRegression"]
SEARCH_FEATURE_STRATEGIES = ["Auto_Select"]
SEARCH_TOP_K = [10, 15, 20, 25, 30, 40, 50]

# ── Elo Settings ───────────────────────────────────────────
ELO_BASE = 1500
ELO_K = 20

# ── Feature Whitelists ────────────────────────────────────
DIFF_FEATURES_WHITELIST = [
    "ROLL_PTS_AVG", "ROLL_AST", "ROLL_REB", "ROLL_STL", "ROLL_BLK",
    "ROLL_TOV", "ROLL_PF", "ROLL_FG_PCT", "ROLL_FG3_PCT",
    "ROLL_FT_PCT", "ROLL_WIN_PCT", "REST_DAYS", "ROLL_WIN_RATE",
    "ROLL_EFG_PCT", "ROLL_TOV_PCT", "ROLL_FT_RATE", "ROLL_ORB"
]
FOUR_FACTORS_LIST = ["ROLL_EFG_PCT", "ROLL_TOV_PCT", "ROLL_FT_RATE", "ROLL_ORB"]

# ── Agentic Behaviour ─────────────────────────────────────
#  The agent runs in phases.  Each phase has its own Optuna
#  study, trial budget, and patience counter.
#
#  Phase 1 (EXPLORE) — wide search to map the landscape
#  Phase 2 (EXPLOIT) — policy-narrowed search for refinement
#  Diagnostics       — triggered when the agent is stuck

PHASE1_TRIALS = 60             # Exploration budget (round-robin → ~20 per model)
PHASE2_TRIALS = 60             # Exploitation budget
PHASE_PATIENCE = 25            # Per-phase patience
REFLECTION_INTERVAL = 15       # Reflect every N trials inside a phase
MAX_TRIALS = 500               # Global safety cap (across all phases)

MEMORY_FILE = os.path.join(OUTPUTS_DIR, "agent_memory.json")
POLICY_FILE = os.path.join(OUTPUTS_DIR, "policy.json")
DIAGNOSTICS_FILE = os.path.join(OUTPUTS_DIR, "diagnostics.md")
PREDICTIONS_FILE = os.path.join(OUTPUTS_DIR, "predictions.json")

# ── NBA Season (auto-detected) ────────────────────────────
# NBA season starts in October: Oct 2025 → "2025-26", Feb 2026 → "2025-26"
_now = datetime.now()
_season_start = _now.year if _now.month >= 10 else _now.year - 1
NBA_CURRENT_SEASON = f"{_season_start}-{str(_season_start + 1)[-2:]}"
