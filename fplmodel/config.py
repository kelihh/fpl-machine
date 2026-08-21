"""Model constants and FPL scoring rules.

Everything tunable lives here so the engine stays readable and you can
sweep parameters without touching logic.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
CACHE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- sources
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"

# Prior-season archive (same FPL API fields, just snapshotted).
# Cloned from https://github.com/vaastav/Fantasy-Premier-League
PRIOR_SEASON = "2025-26"
PRIOR_REPO = "https://github.com/vaastav/Fantasy-Premier-League.git"

# ---------------------------------------------------------------- scoring
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
SAVES_PER_POINT = 3

# Defensive-contribution thresholds. VERIFY these against the current
# rules each season -- they changed when DefCon was introduced and the
# engine is sensitive to them.
DEFCON_THRESHOLD = {"GK": 999, "DEF": 10, "MID": 12, "FWD": 12}
DEFCON_POINTS = 2

# ---------------------------------------------------------------- model
# Flat home/away adjustment applied at the end, matching the historical
# split better than adjusting xG and xGC separately.
HOME_BOOST = 1.05
AWAY_PENALTY = 0.95

# League baseline goals per team per match. Used to put FPL's ordinal
# strength ratings onto a goals scale.
LEAGUE_GOALS_PER_TEAM = 1.42

# Form weighting once the season is underway.
LONG_FORM_WEIGHT = 0.80
SHORT_FORM_WEIGHT = 0.20
SHORT_FORM_GAMES = 6

# Shrinkage toward positional means for players with few prior minutes.
# lambda = minutes / (minutes + PRIOR_STRENGTH)
PRIOR_STRENGTH = 900.0

# How much to trust prior-season xG-derived team strength vs FPL's own
# strength ratings. Promoted sides get 0.0 automatically (no prior data).
XG_PRIOR_WEIGHT = 0.65

# ---------------------------------------------------------------- squad
BUDGET = 1000  # tenths of a million
SQUAD_SIZE = 15
POS_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
MAX_PER_CLUB = 3

# Bench players contribute little; weight them low rather than zero so
# the solver still prefers a bench that might actually play.
BENCH_WEIGHT = [0.20, 0.10, 0.05, 0.02]
