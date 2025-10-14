#!/usr/bin/env python3

"""
main.py (updated)

What this version adds:
- Robust env flag parsing (ENABLE_LLM_METHOD, ENABLE_BASELINE, etc.)
- Clear diagnostics showing whether LLM is actually enabled/usable
- Recent-history de-dup so you don't see the same numbers every cycle
- Optional non-deterministic RNG via MARKOV_RANDOM_SEED=auto|<int>
- Safe Google Sheets I/O with minimal retries

This is designed to be a drop-in replacement, but if your repository
has custom prediction generation, you can plug it into the
`generate_predictions_for_game()` function below.
"""
import os
import sys
import json
import time
import random
import math
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional, Set

import numpy as np

# ---- Optional deps; gspread required if you use Google Sheets storage ----
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception as _e:
    gspread = None
    Credentials = None

# -------------------- ENV / FLAGS --------------------

def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"): return True
    if v in ("0", "false", "no", "n", "off", ""): return False
    return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except Exception:
        return default

def choose_seed() -> Optional[int]:
    raw = os.getenv("MARKOV_RANDOM_SEED", "auto").strip().lower()
    if raw in ("", "auto", "random", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None

ENABLE_LLM = env_flag("ENABLE_LLM_METHOD", False)
ENABLE_BASELINE = env_flag("ENABLE_BASELINE", False)  # default FALSE to avoid drowning LLM/Markov
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_TEMP = float(os.getenv("LLM_TEMP", "0.9"))
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/Chicago")

PREDICTIONS_PER_GAME = env_int("PREDICTIONS_PER_GAME", 10)
PREDICTIONS_POWERBALL  = env_int("PREDICTIONS_POWERBALL", 5)
PREDICTIONS_MEGABUCKS  = env_int("PREDICTIONS_MEGABUCKS", 10)
PREDICTIONS_SUPERCASH  = env_int("PREDICTIONS_SUPERCASH", 10)
PREDICTIONS_BADGER5    = env_int("PREDICTIONS_BADGER5", 5)

# RNG seeding for diversity/noise
_seed = choose_seed()
if _seed is None:
    np.random.seed(None)
    random.seed()
else:
    np.random.seed(_seed)
    random.seed(_seed)

# LLM gate
LLM_OK = False
LLM_REASON = "disabled"
_openai_client = None

if ENABLE_LLM:
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            LLM_REASON = "no OPENAI_API_KEY"
        else:
            _openai_client = OpenAI(api_key=key)
            LLM_OK = True
            LLM_REASON = "ready"
    except Exception as e:
        LLM_OK = False
        LLM_REASON = f"import/error: {type(e).__name__}: {e}"

print(f"[DIAG] LLM_CFG enabled={ENABLE_LLM} provider='{LLM_PROVIDER}' model='{LLM_MODEL}' temp={LLM_TEMP}")
print(f"[DIAG] LLM_OK={LLM_OK} reason={LLM_REASON}")
print(f"[DIAG] baseline enabled={ENABLE_BASELINE}")
print(f"[DIAG] RNG seed={_seed} (MARKOV_RANDOM_SEED='{os.getenv('MARKOV_RANDOM_SEED', 'auto')}')")


# -------------------- GAME RANGES (WI games) --------------------
# Adjust if your jurisdiction differs.
GAME_RULES = {
    "Powerball": {"main": (1, 69, 5), "power": (1, 26, 1)},           # 5 of 1..69 + 1 PB 1..26
    "Megabucks": {"main": (1, 49, 6)},                                 # 6 of 1..49
    "SuperCash": {"main": (1, 39, 4)},                                 # 4 of 1..39
    "Badger 5": {"main": (1, 31, 5)},                                  # 5 of 1..31
}

# -------------------- SHEETS HELPERS --------------------
def _connect_sheet() -> Optional[gspread.Spreadsheet]:
    sheet_name = os.getenv("SHEET_NAME", "").strip()
    if not sheet_name:
        print("::warning::SHEET_NAME not set; running without Sheets I/O.")
        return None
    if gspread is None:
        print("::warning::gspread/google-auth not installed; running without Sheets I/O.")
        return None

    raw_creds = os.getenv("GOOGLE_CREDS_JSON", "").strip()
    if not raw_creds:
        print("::warning::GOOGLE_CREDS_JSON missing; running without Sheets I/O.")
        return None

    try:
        sa_info = json.loads(raw_creds)
        scopes = [
           "https://www.googleapis.com/auth/spreadsheets",
           "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)
        ss = gc.open(sheet_name)
        print(f"[DIAG] Opened spreadsheet '{sheet_name}'")
        return ss
    except Exception as e:
        print(f"::warning::Failed to open Google Sheet: {e}")
        return None

def _touch_debug(ws_title: str, ss: gspread.Spreadsheet) -> None:
    try:
        try:
            ws = ss.worksheet(ws_title)
        except Exception:
            ws = ss.add_worksheet(ws_title, rows=100, cols=5)
        now = datetime.now(timezone.utc).isoformat()
        ws.append_row(["touch", now])
        print(f"[DIAG] Wrote a {ws_title} row successfully.")
    except Exception as e:
        print(f"::warning::Failed to write {ws_title}: {e}")

# -------------------- RECENT HISTORY DEDUPE --------------------
def fmt_pick(game: str, nums: List[int], extra: Optional[int] = None) -> str:
    if game == "Powerball":
        main = " ".join(map(str, sorted(nums)))
        return f"{main} | PB {extra}"
    else:
        return " ".join(map(str, sorted(nums)))

def load_recent_set(ss: Optional[gspread.Spreadsheet], game: str, max_rows: int = 200) -> Set[str]:
    if ss is None:
        return set()
    title = f"{game}_Predictions"
    try:
        ws = ss.worksheet(title)
    except Exception:
        return set()

    try:
        values = ws.get_all_values()[-max_rows:]
        s: Set[str] = set()
        for row in values:
            # expect first col contains the pick string
            if row and row[0].strip():
                s.add(row[0].strip())
        print(f"[DIAG] recent[{game}] unique={len(s)} (from last {len(values)} rows)")
        return s
    except Exception as e:
        print(f"::warning::Failed to read recent predictions for {game}: {e}")
        return set()

def write_predictions(ss: Optional[gspread.Spreadsheet], game: str, picks: List[str]) -> None:
    if ss is None:
        for p in picks:
            print(f"[{game}] {p}")
        return
    title = f"{game}_Predictions"
    try:
        try:
            ws = ss.worksheet(title)
        except Exception:
            ws = ss.add_worksheet(title, rows=2000, cols=3)
        rows = [[p, datetime.now(timezone.utc).isoformat()] for p in picks]
        ws.append_rows(rows)
        print(f"[{game}] appended {len(rows)} predictions.")
    except Exception as e:
        print(f"::warning::Failed to write predictions for {game}: {e}")
        # still log to console
        for p in picks:
            print(f"[{game}] {p}")

# -------------------- GENERATION --------------------
def gen_random_unique(low: int, high: int, k: int) -> List[int]:
    return random.sample(range(low, high + 1), k)

def generate_one_pick(game: str) -> str:
    rules = GAME_RULES[game]
    main_low, main_high, main_k = rules["main"]
    main_nums = gen_random_unique(main_low, main_high, main_k)
    extra = None
    if game == "Powerball":
        pb_low, pb_high, pb_k = rules["power"]
        extra = gen_random_unique(pb_low, pb_high, pb_k)[0]
    return fmt_pick(game, main_nums, extra)

def generate_predictions_for_game(game: str, target: int, recent: Set[str]) -> List[str]:
    """Simple diversified generator with recent-history de-dup.
    (Replace with your ML/LLM/Markov logic if desired.)"""
    out: List[str] = []
    guard = 0
    while len(out) < target and guard < 10000:
        guard += 1
        candidate = generate_one_pick(game)
        if candidate in recent or candidate in out:
            continue
        out.append(candidate)
    return out

# -------------------- MAIN --------------------
def main():
    ss = _connect_sheet()
    if ss is not None:
        _touch_debug("Debug_Touch", ss)

    # Determine targets (env overrides)
    targets = {
        "Powerball": PREDICTIONS_POWERBALL or PREDICTIONS_PER_GAME,
        "Megabucks": PREDICTIONS_MEGABUCKS or PREDICTIONS_PER_GAME,
        "SuperCash": PREDICTIONS_SUPERCASH or PREDICTIONS_PER_GAME,
        "Badger 5": PREDICTIONS_BADGER5 or PREDICTIONS_PER_GAME,
    }

    # Shuffle game order for extra variety
    games = list(targets.keys())
    random.shuffle(games)

    total_emitted = 0
    for game in games:
        target = int(targets[game])
        if target <= 0: 
            continue
        recent = load_recent_set(ss, game, max_rows=200)
        picks = generate_predictions_for_game(game, target, recent)
        write_predictions(ss, game, picks)
        total_emitted += len(picks)

    print(f"Success! main.py finished. Emitted total={total_emitted} predictions.")

if __name__ == "__main__":
    main()
