# main.py
import os
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import gspread  # already in your env
from google.oauth2.service_account import Credentials

# ---------------------------
# Config / CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Override 'only after new draw'. If not set, predicts only on new results.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run logic without writing predictions or updating state.")
    return p.parse_args()

SHEET_NAME = os.getenv("SHEET_NAME", "Lottery Predictor New August 25")

# Fallback counts (override these with your existing variables or Config tab)
DEFAULT_COUNTS = {
    "Powerball": int(os.getenv("PRED_COUNT_POWERBALL", "20")),
    "Megabucks": int(os.getenv("PRED_COUNT_MEGABUCKS", "20")),
    "Super Cash": int(os.getenv("PRED_COUNT_SUPERCASH", "20")),
    "Badger 5": int(os.getenv("PRED_COUNT_BADGER5", "20")),
}

ENABLE_PREDICTIONS = os.getenv("ENABLE_PREDICTIONS", "").lower() in {"1", "true", "yes"}

# ---------------------------
# Google Sheets helpers
# ---------------------------
def open_sheet() -> gspread.Spreadsheet:
    # Expect GOOGLE_CREDS_JSON in env (as you already have)
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)  # secrets.GOOGLE_CREDS_JSON is a JSON string
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open(SHEET_NAME)

def get_or_create_ws(ss: gspread.Spreadsheet, title: str, header: List[str]) -> gspread.Worksheet:
    try:
        ws = ss.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=title, rows="1000", cols=str(len(header)+5))
        ws.append_row(header, value_input_option="RAW")
    return ws

# ---------------------------
# State tracking
# ---------------------------
STATE_HEADERS = ["Game", "LastProcessedKey", "UpdatedAtUTC"]

def read_state_map(state_ws) -> Dict[str, str]:
    data = state_ws.get_all_values()
    if not data:
        return {}
    header = {name: idx for idx, name in enumerate(data[0])}
    out = {}
    for row in data[1:]:
        if not row or len(row) < 2:
            continue
        game = row[header["Game"]].strip()
        key  = row[header["LastProcessedKey"]].strip()
        if game:
            out[game] = key
    return out

def write_state_key(state_ws, game: str, batch_key: str):
    data = state_ws.get_all_values() or [STATE_HEADERS]
    header = {name: idx for idx, name in enumerate(data[0])}
    # find row for game
    row_idx = None
    for i, row in enumerate(data[1:], start=2):
        if row and len(row) > header["Game"] and row[header["Game"]].strip() == game:
            row_idx = i
            break
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if row_idx is None:
        # append
        state_ws.append_row([game, batch_key, now_utc], value_input_option="RAW")
    else:
        # update in place
        state_ws.update_cell(row_idx, header["LastProcessedKey"]+1, batch_key)
        state_ws.update_cell(row_idx, header["UpdatedAtUTC"]+1, now_utc)

# ---------------------------
# Results & predictions I/O
# ---------------------------
# TODO: Map these to your sheets. Examples assume a single “Results” tab with columns:
# ["Game","DrawDate","DrawID","Numbers"]
RESULTS_HEADERS = ["Game", "DrawDate", "DrawID", "Numbers"]

def latest_result_for_game(results_ws, game: str) -> Optional[Dict[str, Any]]:
    data = results_ws.get_all_values()
    if not data or data[0][:4] != RESULTS_HEADERS:
        raise RuntimeError(
            f"'Results' sheet must have headers {RESULTS_HEADERS}. "
            f"Adjust latest_result_for_game() to your schema."
        )
    header = {name: idx for idx, name in enumerate(data[0])}
    latest: Optional[Tuple[str, Dict[str, Any]]] = None  # (draw_date_iso, row_dict)

    for row in data[1:]:
        if not row or len(row) < 4:
            continue
        if row[header["Game"]].strip() != game:
            continue
        draw_date = row[header["DrawDate"]].strip()  # expect ISO-like or consistent sortable
        draw_id   = row[header["DrawID"]].strip()
        numbers   = row[header["Numbers"]].strip()
        rowd = {"Game": game, "DrawDate": draw_date, "DrawID": draw_id, "Numbers": numbers}
        key_date = draw_date  # must be lexicographically comparable; if not, parse to dt
        if latest is None or key_date > latest[0]:
            latest = (key_date, rowd)
    return latest[1] if latest else None

# Predictions sheet structure: one row per prediction with a "BatchKey" to bind them together.
PRED_HEADERS = ["TimestampUTC","Game","BatchKey","Method","Params","Pick"]

def already_predicted_for_batch(pred_ws, game: str, batch_key: str) -> bool:
    # Efficient check: filter by game and batch key using a single read.
    data = pred_ws.get_all_values()
    if not data:
        return False
    header = {name: idx for idx, name in enumerate(data[0])}
    gi = header.get("Game"); bi = header.get("BatchKey")
    if gi is None or bi is None:
        # Sheet missing expected columns -> treat as not predicted (we'll append with proper headers)
        return False
    for row in data[1:]:
        if not row or len(row) <= bi:
            continue
        if row[gi] == game and row[bi] == batch_key:
            return True
    return False

def write_prediction_rows(pred_ws, rows: List[List[Any]]):
    if not rows:
        return
    pred_ws.append_rows(rows, value_input_option="RAW")

# ---------------------------
# Prediction generator
# ---------------------------
# Plug in your actual number ranges and constraints per game below.
from random import sample, randint

GAME_RANGES = {
    # Example ranges; replace with your actual
    "Powerball": {"main_count": 5, "main_min": 1, "main_max": 69, "bonus_count": 1, "bonus_min": 1, "bonus_max": 26},
    "Megabucks": {"main_count": 6, "main_min": 1, "main_max": 49, "bonus_count": 0},
    "Super Cash": {"main_count": 6, "main_min": 1, "main_max": 39, "bonus_count": 0},
    "Badger 5": {"main_count": 5, "main_min": 1, "main_max": 31, "bonus_count": 0},
}

def draw_candidate(game: str) -> List[int]:
    cfg = GAME_RANGES[game]
    mains = sorted(sample(range(cfg["main_min"], cfg["main_max"]+1), cfg["main_count"]))
    if cfg.get("bonus_count", 0):
        bonus = [randint(cfg["bonus_min"], cfg["bonus_max"])]
        return mains + bonus
    return mains

def passes_constraints(game: str, pick: List[int], strict_level: int, recent_draws: List[List[int]]) -> bool:
    """
    strict_level: 0 = strictest, higher = more relaxed.
    Example constraints to illustrate pattern—replace with yours:
      - No exact match to any of the last D draws (D reduces as strict_level rises)
      - Sum within min/max (bounds widen as strict_level rises)
    """
    # Example recent no-duplicate constraint
    base_no_dup_depth = 8
    depth = max(0, base_no_dup_depth - strict_level*2)
    for past in recent_draws[:depth]:
        if pick == past:
            return False

    # Example sum bounds that widen
    s = sum(pick)
    base_lo, base_hi = 60, 220
    wiggle = strict_level * 10
    if not (base_lo - wiggle <= s <= base_hi + wiggle):
        return False

    return True

def generate_batch(game: str, target: int, recent_draws: List[List[int]]) -> Tuple[List[List[int]], int]:
    """
    Try strict -> relax until we hit 'target'.
    Returns (picks, final_strict_level_used).
    """
    picks: List[List[int]] = []
    seen = set()
    strict_level = 0
    MAX_STRICT_LEVEL = 6
    MAX_ATTEMPTS_PER_LEVEL = 4000

    while len(picks) < target:
        attempts = 0
        added_this_level = 0
        while len(picks) < target and attempts < MAX_ATTEMPTS_PER_LEVEL:
            c = draw_candidate(game)
            t = tuple(c)
            attempts += 1
            if t in seen:
                continue
            if passes_constraints(game, c, strict_level, recent_draws):
                picks.append(c)
                seen.add(t)
                added_this_level += 1
        if len(picks) < target:
            if strict_level >= MAX_STRICT_LEVEL:
                # Last resort: allow near-duplicates within this batch, but log it
                while len(picks) < target:
                    c = draw_candidate(game)
                    picks.append(c)
                break
            strict_level += 1
    return picks, strict_level

# ---------------------------
# Utilities
# ---------------------------
def canonical_batch_key(draw_date: str, draw_id: str) -> str:
    draw_id = draw_id or ""
    return f"{draw_date}#{draw_id}"

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# Main Run
# ---------------------------
def run():
    args = get_args()
    force = bool(args.force)

    if not ENABLE_PREDICTIONS and not force:
        print("ENABLE_PREDICTIONS is false; exiting.")
        return

    ss = open_sheet()
    state_ws = get_or_create_ws(ss, "State", STATE_HEADERS)
    pred_ws  = get_or_create_ws(ss, "Predictions", PRED_HEADERS)
    results_ws = get_or_create_ws(ss, "Results", RESULTS_HEADERS)  # assumes single results tab

    state_map = read_state_map(state_ws)

    # TODO: Replace this list with your config source if you have a "Games" sheet.
    games = list(DEFAULT_COUNTS.keys())

    for game in games:
        target = DEFAULT_COUNTS[game]

        latest = latest_result_for_game(results_ws, game)
        if not latest:
            print(f"[{game}] No results found; skipping.")
            continue

        draw_date = latest["DrawDate"]
        draw_id   = latest.get("DrawID", "")
        batch_key = canonical_batch_key(draw_date, draw_id)

        last_key  = state_map.get(game, "")
        is_new = (batch_key != last_key)

        print(f"[{game}] latest={batch_key} last={last_key or '—'} is_new={is_new} force={force}")

        if not is_new and not force:
            print(f"[{game}] No new draw and --force not set; skipping predictions.")
            continue

        if already_predicted_for_batch(pred_ws, game, batch_key) and not force:
            print(f"[{game}] Predictions already exist for batch {batch_key}; skipping.")
            # Still update state to prevent re-runs if state was behind.
            if not args.dry_run:
                write_state_key(state_ws, game, batch_key)
            continue

        # TODO: Fetch recent actual draws for constraints (for your 'no duplicates' etc.)
        # If you have them in Results, read the last N rows for this game and parse "Numbers".
        recent_draws: List[List[int]] = []  # plug your parser here

        picks, level = generate_batch(game, target, recent_draws)
        if len(picks) != target:
            print(f"[{game}] WARN: generated {len(picks)} != target {target} after relax level {level}")

        ts = now_utc_str()
        rows = [[ts, game, batch_key, "v4.2.9", json.dumps({"level": level}), " ".join(map(str, p))] for p in picks]

        print(f"[{game}] Writing {len(rows)} predictions for batch {batch_key} (level={level})")
        if not args.dry_run:
            write_prediction_rows(pred_ws, rows)
            write_state_key(state_ws, game, batch_key)

    print("Success! Finished.")

if __name__ == "__main__":
    run()
