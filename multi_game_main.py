"""Multi-game lottery predictor adapted for Lottery Predictor New August 25.

This script generates Markov-based lottery predictions and writes them into a shared
Prediction_Tracker tab.  It has been tailored to match the specific Google Sheet
structure found at:

  https://docs.google.com/spreadsheets/d/1AeumsjywQ2pECb34eR0_ms8yH6D9mjb4Gb9bhRQ9NuU/edit

Key features and modifications:

* **Per-game configuration**: Each game is defined with a result tab name, number range
  and whether it includes a bonus ball.  The built-in mapping below reflects the
  Wisconsin games present in the sheet (`Badger 5`, `Super Cash!`, `Megabucks`,
  and `Powerball`).  For Powerball, the red Powerball is treated separately with
  its own range (1..26).

* **Flexible parsing**: Historical results are read from the `_Results` tabs.  The
  parser reads numeric values to the right of the date until encountering a blank
  or non-numeric cell.  For Powerball this picks up both the five white balls
  and the red Powerball; the latter is then separated as the bonus.

* **Markov prediction**: A first‑order Markov chain is trained on the main numbers
  for each position.  Predictions are sampled using softmax with temperature and
  Laplace smoothing.  A simple constraint rejects sets with long runs of consecutive
  integers.

* **Bonus ball handling**: For games with a bonus ball (currently Powerball) a
  separate chain over the bonus values is trained and sampled.

* **Auto-scoring**: When a new official result appears, unscored predictions in
  `Prediction_Tracker` with matching `Next Draw Date` are scored.  For bonus games
  the bonus must also match to count as a win.

* **Next draw date**: The script attempts to read a `Games_Index` tab with
  `Game` and `NextDrawDate` columns.  If this tab or entry is missing, the
  next draw date is left blank.

Environment variables:

  GCP_SERVICE_ACCOUNT_JSON : JSON credentials for a Google service account (required).
  GOOGLE_SHEET_ID          : ID of the target Google Sheet (required).
  GOOGLE_SHEET_NAME        : Optional friendly name of the sheet.
  GAMES_LIST               : Comma-separated list of game names to process (e.g.
                              "Badger 5,Super Cash,Megabucks,Powerball").
  PREDICTIONS_PER_RUN      : Number of predictions to generate per game (default 20).
  LAPLACE_ALPHA            : Laplace smoothing parameter (default 0.5).
  SOFTMAX_TAU              : Temperature for softmax sampling (default 1.0).
  NO_CONSEC_LIMIT          : Reject sets with runs of this length or more (default 3).

Per-game overrides can be set using environment variables.  The game name is
uppercased and spaces replaced with underscores.  For example, for "Badger 5"
use `BADGER_5_NUM_MAIN_BALLS`, `BADGER_5_MAIN_MIN`, `BADGER_5_MAIN_MAX`.  For
games with a bonus ball (e.g. Powerball) also set `POWERBALL_BONUS_COUNT`,
`POWERBALL_BONUS_MIN`, and `POWERBALL_BONUS_MAX`.
"""

import os
import json
import random
import math
import re
from typing import List, Dict, Tuple, Any
import datetime as dt

import pytz
import gspread
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------------------------
# Configuration

# Spreadsheet identity and timezone
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
LOCAL_TZ = pytz.timezone(os.getenv("LOCAL_TZ", "America/Chicago"))

# Games to process (comma-separated names).  Game names should match the keys
# in KNOWN_GAMES below, but are compared case-insensitively.
GAMES_LIST = [g.strip() for g in os.getenv("GAMES_LIST", "Badger 5,Super Cash,Megabucks,Powerball").split(",") if g.strip()]

# Number of predictions per game per run
PREDICTIONS_PER_RUN = int(os.getenv("PREDICTIONS_PER_RUN", "20"))

# Markov parameters
LAPLACE_ALPHA = float(os.getenv("LAPLACE_ALPHA", "0.5"))
SOFTMAX_TAU   = float(os.getenv("SOFTMAX_TAU", "1.0"))
NO_CONSEC_LIMIT = int(os.getenv("NO_CONSEC_LIMIT", "3"))

# Built-in mapping for known games.  Each entry contains:
#   tab        : Name of the worksheet containing results.
#   need       : Number of main numbers to draw.
#   lo, hi     : Inclusive bounds for the main numbers.
#   has_bonus  : True if the game has a separate bonus ball.
#   bonus_count: Number of bonus balls (currently only 1 supported).
#   bonus_lo, bonus_hi : Range of the bonus ball(s), if applicable.
KNOWN_GAMES: Dict[str, Dict[str, Any]] = {
    # Badger 5: five numbers from 1..31
    "badger 5": {
        "tab": "Badger5_Results",
        "need": 5,
        "lo": 1,
        "hi": 31,
        "has_bonus": False,
    },
    # Super Cash!: six numbers from 1..39 (no bonus)
    "super cash": {
        "tab": "SuperCash_Results",
        "need": 6,
        "lo": 1,
        "hi": 39,
        "has_bonus": False,
    },
    # Megabucks: six numbers from 1..49 (no bonus)
    "megabucks": {
        "tab": "Megabucks_Results",
        "need": 6,
        "lo": 1,
        "hi": 49,
        "has_bonus": False,
    },
    # Powerball: five white balls from 1..69 and one Powerball from 1..26
    "powerball": {
        "tab": "Powerball_Results",
        "need": 5,
        "lo": 1,
        "hi": 69,
        "has_bonus": True,
        "bonus_count": 1,
        "bonus_lo": int(os.getenv("POWERBALL_BONUS_MIN", "1")),
        "bonus_hi": int(os.getenv("POWERBALL_BONUS_MAX", "26")),
    },
}


def override_from_env(game_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of cfg overridden by environment variables for game_name."""
    cfg = dict(cfg)  # shallow copy
    key = game_name.upper().replace(" ", "_")
    # Override main ball parameters
    nm = os.getenv(f"{key}_NUM_MAIN_BALLS")
    lo = os.getenv(f"{key}_MAIN_MIN")
    hi = os.getenv(f"{key}_MAIN_MAX")
    if nm:
        try:
            cfg["need"] = int(nm)
        except Exception:
            pass
    if lo:
        try:
            cfg["lo"] = int(lo)
        except Exception:
            pass
    if hi:
        try:
            cfg["hi"] = int(hi)
        except Exception:
            pass
    # Override tab
    tab_override = os.getenv(f"{key}_RESULTS_TAB")
    if tab_override:
        cfg["tab"] = tab_override
    # Bonus overrides, if applicable
    if cfg.get("has_bonus"):
        bc  = os.getenv(f"{key}_BONUS_COUNT")
        blo = os.getenv(f"{key}_BONUS_MIN")
        bhi = os.getenv(f"{key}_BONUS_MAX")
        if bc:
            try:
                cfg["bonus_count"] = int(bc)
            except Exception:
                pass
        if blo:
            try:
                cfg["bonus_lo"] = int(blo)
            except Exception:
                pass
        if bhi:
            try:
                cfg["bonus_hi"] = int(bhi)
            except Exception:
                pass
    return cfg


# ---------------------------------------------------------------------------
# Google Sheets helpers

def open_sheet() -> gspread.Spreadsheet:
    """Authenticate using the service account JSON and open the spreadsheet."""
    if not SHEET_ID:
        raise EnvironmentError("GOOGLE_SHEET_ID must be set")
    json_creds = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not json_creds:
        raise EnvironmentError("GCP_SERVICE_ACCOUNT_JSON must be set")
    info = json.loads(json_creds)
    creds = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(SHEET_ID)


def get_or_create(sheet: gspread.Spreadsheet, title: str, rows: int = 20000, cols: int = 20) -> gspread.Worksheet:
    """Return a worksheet with the given title, creating it if it does not exist."""
    try:
        return sheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return sheet.add_worksheet(title=title, rows=rows, cols=cols)


def now_local_str() -> str:
    """Return current time in LOCAL_TZ as an ISO-like string."""
    return dt.datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def lookup_next_draw_date(sheet: gspread.Spreadsheet, game_name: str) -> str:
    """Look up the next draw date for a game from the `Games_Index` tab, if present."""
    try:
        w = sheet.worksheet("Games_Index")
    except gspread.WorksheetNotFound:
        return ""
    vals = w.get_all_values()
    if not vals or len(vals) < 2:
        return ""
    for r in vals[1:]:
        if not r:
            continue
        g = (r[0] or "").strip().lower()
        nd = (r[1] or "").strip() if len(r) > 1 else ""
        if g == game_name.strip().lower() and nd:
            return nd
    return ""


# ---------------------------------------------------------------------------
# Data loading and parsing

def parse_row_numbers(row: List[str], expected_need: int) -> Tuple[List[int], List[int]]:
    """Parse numeric values from a result row.

    The row is expected to have the draw date in column 0 and numbers in subsequent
    columns.  This function reads numeric values until a blank or non-numeric cell.
    Returns a tuple (mains, bonus), where `mains` contains the first `expected_need`
    numbers and `bonus` contains any subsequent numbers.
    """
    nums: List[int] = []
    for cell in row[1:]:
        s = (cell or "").strip()
        if not s:
            break
        try:
            nums.append(int(s))
        except Exception:
            break
    mains = nums[:expected_need]
    bonus = nums[expected_need:]
    return mains, bonus


def load_history(sheet: gspread.Spreadsheet, tab: str, need: int, has_bonus: bool, bonus_count: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Load historical draws for a game from the specified tab.

    Returns (mains_history, bonus_history) as lists of lists.  The latest draw
    should be first (top) in the sheet.  For games without a bonus, bonus_history
    is an empty list.
    """
    ws = get_or_create(sheet, tab)
    rows = ws.get_all_values()
    if not rows or len(rows) < 2:
        return [], []
    mains_hist: List[List[int]] = []
    bonus_hist: List[List[int]] = []
    data = rows[1:]  # skip header
    for r in data:
        mains, bonus = parse_row_numbers(r, need)
        if len(mains) == need:
            mains_hist.append(mains)
            if has_bonus and bonus and bonus_count > 0:
                bonus_hist.append(bonus[:bonus_count])
    return mains_hist, bonus_hist


def load_latest_result(sheet: gspread.Spreadsheet, tab: str, need: int, has_bonus: bool, bonus_count: int) -> Tuple[str, List[int], List[int]]:
    """Return the latest result as (draw_date, mains, bonus).

    If no data is available, returns ("", [], []).
    """
    ws = get_or_create(sheet, tab)
    rows = ws.get_all_values()
    if not rows or len(rows) < 2:
        return "", [], []
    latest = rows[1]
    draw_date = (latest[0] or "").strip()
    mains, bonus = parse_row_numbers(latest, need)
    if has_bonus and bonus_count > 0:
        return draw_date, mains, bonus[:bonus_count]
    return draw_date, mains, []


# ---------------------------------------------------------------------------
# Markov training and sampling

def train_markov_positionwise(history: List[List[int]], need: int) -> List[Dict[int, Dict[int, int]]]:
    """Train a separate first-order chain for each position in the draw."""
    counts: List[Dict[int, Dict[int, int]]] = [dict() for _ in range(need)]
    # chronological order (oldest first) for transitions
    hist = list(reversed(history))
    for a_row, b_row in zip(hist, hist[1:]):
        for pos in range(need):
            a, b = a_row[pos], b_row[pos]
            d = counts[pos].setdefault(a, {})
            d[b] = d.get(b, 0) + 1
    return counts


def next_distribution(counts_for_pos: Dict[int, Dict[int, int]], prev: int, lo: int, hi: int, alpha: float) -> Dict[int, float]:
    """Compute the smoothed distribution of next values for a given position."""
    row = counts_for_pos.get(prev, {})
    denom = sum(row.values()) + alpha * (hi - lo + 1)
    return {x: (row.get(x, 0) + alpha) / denom for x in range(lo, hi + 1)}


def softmax(weights: Dict[int, float], tau: float) -> List[Tuple[int, float]]:
    """Apply softmax with temperature to a dictionary of weights."""
    if not weights:
        return []
    mx = max(weights.values()) if weights else 0.0
    scores: Dict[int, float] = {k: math.exp((v - mx) / max(tau, 1e-6)) for k, v in weights.items()}
    s = sum(scores.values()) or 1.0
    return [(k, scores[k] / s) for k in scores]


def categorical_sample(dist: List[Tuple[int, float]]) -> int:
    """Sample a value from a list of (value, probability) pairs."""
    r = random.random()
    cum = 0.0
    for k, p in dist:
        cum += p
        if r <= cum:
            return k
    return dist[-1][0]


def too_many_consecutives(nums: List[int], limit: int) -> bool:
    """Return True if nums contains a run of >= limit consecutive integers."""
    if limit <= 1:
        return False
    s = set(nums)
    longest = 1
    for n in nums:
        if (n - 1) not in s:
            length = 1
            while (n + length) in s:
                length += 1
            if length > longest:
                longest = length
            if longest >= limit:
                return True
    return False


def generate_mains(history: List[List[int]], need: int, lo: int, hi: int) -> List[int]:
    """Generate a main-number prediction via a position-wise Markov chain."""
    if not history:
        s: set[int] = set()
        while len(s) < need:
            s.add(random.randint(lo, hi))
        return sorted(s)
    counts = train_markov_positionwise(history, need)
    seed_row = history[0]
    picks: List[int] = []
    used: set[int] = set()
    for pos in range(need):
        prev = seed_row[pos]
        dist_raw = next_distribution(counts[pos], prev, lo, hi, LAPLACE_ALPHA)
        # Exclude already chosen numbers
        for u in list(used):
            if u in dist_raw:
                dist_raw[u] = 0.0
        dist = softmax(dist_raw, SOFTMAX_TAU)
        # normalise after zeroing
        Z = sum(p for _, p in dist) or 1.0
        dist = [(k, p / Z) for k, p in dist]
        choice = categorical_sample(dist)
        picks.append(choice)
        used.add(choice)
    # enforce no long consecutive runs
    if NO_CONSEC_LIMIT and too_many_consecutives(picks, NO_CONSEC_LIMIT):
        return generate_mains(history, need, lo, hi)
    return sorted(picks)


def train_markov_single(seq: List[List[int]]) -> Dict[int, Dict[int, int]]:
    """Train a first-order Markov chain on a single-valued history (e.g., bonus)."""
    counts: Dict[int, Dict[int, int]] = {}
    # Flatten to a simple list
    vals = [row[0] for row in reversed(seq)] if seq else []
    for a, b in zip(vals, vals[1:]):
        d = counts.setdefault(a, {})
        d[b] = d.get(b, 0) + 1
    return counts


def next_distribution_single(counts: Dict[int, Dict[int, int]], prev: int, lo: int, hi: int) -> Dict[int, float]:
    """Next-value distribution for a single-variable Markov chain."""
    row = counts.get(prev, {})
    denom = sum(row.values()) + LAPLACE_ALPHA * (hi - lo + 1)
    return {x: (row.get(x, 0) + LAPLACE_ALPHA) / denom for x in range(lo, hi + 1)}


def generate_bonus(history_bonus: List[List[int]], lo: int, hi: int) -> int:
    """Generate a bonus ball using a first-order Markov chain over bonus values."""
    if not history_bonus:
        return random.randint(lo, hi)
    counts = train_markov_single(history_bonus)
    prev = history_bonus[0][0]
    probs = next_distribution_single(counts, prev, lo, hi)
    dist = softmax(probs, SOFTMAX_TAU)
    Z = sum(p for _, p in dist) or 1.0
    dist = [(k, p / Z) for k, p in dist]
    return categorical_sample(dist)


# ---------------------------------------------------------------------------
# Prediction tracking

def ensure_tracker(sheet: gspread.Spreadsheet) -> gspread.Worksheet:
    """Ensure Prediction_Tracker exists and has the correct header."""
    ws = get_or_create(sheet, "Prediction_Tracker", rows=60000, cols=12)
    vals = ws.get_all_values()
    if not vals:
        ws.update(
            [["Timestamp", "Game", "Next Draw Date", "Prediction", "Method", "Win Count", "Matches", "Match Count"]],
            "A1",
        )
    return ws


def fmt_prediction_mains(mains: List[int]) -> str:
    return " ".join(f"{n:02d}" for n in sorted(mains))


def fmt_prediction_powerball(mains: List[int], bonus: int) -> str:
    return f"{fmt_prediction_mains(mains)} | PB {bonus:02d}"


def parse_pred_str(pred: str, need: int, bonus_count: int) -> Tuple[List[int], List[int]]:
    """Parse a prediction string into (mains, bonus) using digit extraction.

    The prediction string may be in formats like:
      '05 12 19 33 48 | PB 09'
      '5 12 19 33 48 9'
      '5-12-19-33-48 +PB 9'

    All numeric tokens are extracted; the first `need` numbers are returned as mains
    and the next `bonus_count` numbers as bonus.
    """
    if not pred:
        return [], []
    # Extract all integer tokens (using regex to handle various separators)
    ints = [int(x) for x in re.findall(r"\d+", pred)]
    mains = ints[:need]
    bonus = ints[need:need + bonus_count] if bonus_count > 0 else []
    return mains, bonus


def overlap(a: List[int], b: List[int]) -> List[int]:
    """Return sorted intersection of two integer lists."""
    return sorted(list(set(a).intersection(b)))


def autoscore_latest_if_ready(
    sheet: gspread.Spreadsheet,
    game_name: str,
    tab: str,
    need: int,
    has_bonus: bool,
    bonus_count: int,
) -> None:
    """Auto-score unscored predictions if the latest result is available.

    For each unscored row in Prediction_Tracker with matching game and next draw date,
    compute the number of matching balls (including bonus where applicable) and update
    Win Count, Matches, and Match Count.
    """
    latest_date, latest_mains, latest_bonus = load_latest_result(sheet, tab, need, has_bonus, bonus_count)
    if not latest_date or len(latest_mains) != need:
        return
    ws = ensure_tracker(sheet)
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return
    header = vals[0]
    data = vals[1:]
    try:
        i_game = header.index("Game")
        i_next = header.index("Next Draw Date")
        i_pred = header.index("Prediction")
        i_method = header.index("Method")
        i_win = header.index("Win Count")
        i_match = header.index("Matches")
        i_mcnt = header.index("Match Count")
    except ValueError:
        return
    updates: List[Tuple[int, int, str, str]] = []
    for idx, row in enumerate(data, start=2):
        try:
            if (row[i_game] or "").strip().lower() != game_name.strip().lower():
                continue
            # Only auto-score rows with same draw date
            if (row[i_next] or "").strip() != latest_date:
                continue
            # Skip already scored
            if (row[i_mcnt] or "").strip():
                continue
            pred_str = row[i_pred] or ""
            pred_mains, pred_bonus = parse_pred_str(pred_str, need, bonus_count)
            # Determine win and matches
            win = 0
            if has_bonus:
                exact_main = sorted(pred_mains) == sorted(latest_mains)
                bonus_match = (pred_bonus[:bonus_count] == latest_bonus[:bonus_count]) if latest_bonus else False
                win = 1 if (exact_main and bonus_match) else 0
                inter = overlap(pred_mains, latest_mains)
                # Build matches string
                match_str = " ".join(f"{n:02d}" for n in inter)
                if bonus_match and latest_bonus:
                    match_str = (match_str + " | PB " + f"{latest_bonus[0]:02d}").strip()
                mcnt = len(inter) + (1 if bonus_match else 0)
            else:
                exact_main = sorted(pred_mains) == sorted(latest_mains)
                win = 1 if exact_main else 0
                inter = overlap(pred_mains, latest_mains)
                match_str = " ".join(f"{n:02d}" for n in inter)
                mcnt = len(inter)
            updates.append((idx, win, match_str, str(mcnt)))
        except Exception:
            continue
    # Apply updates (row by row)
    for row_idx, win, matches, mcnt in updates:
        ws.update_cell(row_idx, i_win + 1, win)
        ws.update_cell(row_idx, i_match + 1, matches)
        ws.update_cell(row_idx, i_mcnt + 1, mcnt)


def append_predictions(sheet: gspread.Spreadsheet, game_name: str, next_draw_date: str, pred_strings: List[str]) -> None:
    """Append new prediction rows to Prediction_Tracker."""
    ws = ensure_tracker(sheet)
    ts = now_local_str()
    rows: List[List[Any]] = []
    for pstr in pred_strings:
        rows.append([ts, game_name, next_draw_date, pstr, "markov_v2", "0", "", ""])
    if rows:
        ws.append_rows(rows, value_input_option="RAW")


# ---------------------------------------------------------------------------
# Main processing per game

def run_game(sheet: gspread.Spreadsheet, game_name: str) -> None:
    """Process a single game: score previous predictions and generate new ones."""
    # Look up base configuration
    base_cfg = KNOWN_GAMES.get(game_name.lower().strip())
    if not base_cfg:
        print(f"Unknown game '{game_name}'. Skipping.")
        return
    cfg = override_from_env(game_name, base_cfg)
    tab = cfg.get("tab")
    need = cfg.get("need")
    lo = cfg.get("lo")
    hi = cfg.get("hi")
    has_bonus = cfg.get("has_bonus", False)
    bonus_count = cfg.get("bonus_count", 0)
    bonus_lo = cfg.get("bonus_lo", None)
    bonus_hi = cfg.get("bonus_hi", None)
    if not tab or need is None or lo is None or hi is None:
        print(f"Incomplete configuration for game '{game_name}'. Skipping.")
        return
    # Score any pending predictions
    autoscore_latest_if_ready(sheet, game_name, tab, need, has_bonus, bonus_count)
    # Load history
    mains_hist, bonus_hist = load_history(sheet, tab, need, has_bonus, bonus_count)
    # Determine next draw date
    next_draw_date = lookup_next_draw_date(sheet, game_name)
    # Generate predictions
    pred_strings: List[str] = []
    for _ in range(max(1, PREDICTIONS_PER_RUN)):
        mains = generate_mains(mains_hist, need, lo, hi)
        if has_bonus and bonus_lo is not None and bonus_hi is not None and bonus_count > 0:
            bonus = generate_bonus(bonus_hist, bonus_lo, bonus_hi)
            pred_strings.append(fmt_prediction_powerball(mains, bonus))
        else:
            pred_strings.append(fmt_prediction_mains(mains))
    # Append predictions
    append_predictions(sheet, game_name, next_draw_date, pred_strings)
    print(f"[{game_name}] wrote {len(pred_strings)} predictions (next draw: {next_draw_date})")


def main() -> None:
    if not SHEET_ID:
        raise EnvironmentError("GOOGLE_SHEET_ID must be set")
    # Authenticate and open sheet
    sheet = open_sheet()
    # Ensure tracker exists
    ensure_tracker(sheet)
    # Iterate games
    for game in GAMES_LIST:
        run_game(sheet, game)


if __name__ == "__main__":
    main()