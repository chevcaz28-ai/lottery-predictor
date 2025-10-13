#!/usr/bin/env python3

"""
markov_predictor.py (updated)

Adds:
- RNG seed handling via MARKOV_RANDOM_SEED
- Simple exponential backoff for Sheets calls (429/5xx)
- Recent-history de-dup (shared behavior with main)
- Temperature parameter (MARKOV_TEMP) controlling small amount of noise

Note: This is a lightweight stand-in if your original file contained a
richer Markov chain model. You can graft your own transition matrix
logic into `markov_generate_one_pick()` while retaining the I/O,
dedupe, and retry scaffolding provided here.
"""
import os
import sys
import json
import time
import math
import random
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime, timezone

import numpy as np

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception as _e:
    gspread = None
    Credentials = None

# -------------------- ENV --------------------
MARKOV_TEMP = float(os.getenv("MARKOV_TEMP", "1.05"))
MARKOV_SMOOTH = float(os.getenv("MARKOV_SMOOTH", "1.0"))
RAW_SEED = os.getenv("MARKOV_RANDOM_SEED", "auto").strip().lower()

def choose_seed(raw: str) -> Optional[int]:
    if raw in ("", "auto", "random", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None

_seed = choose_seed(RAW_SEED)
if _seed is None:
    np.random.seed(None); random.seed()
else:
    np.random.seed(_seed); random.seed(_seed)

print(f"[markov_mc] RNG seed={_seed} (raw='{RAW_SEED}') temp={MARKOV_TEMP} smooth={MARKOV_SMOOTH}")

GAME_RULES = {
    "Powerball": {"main": (1, 69, 5), "power": (1, 26, 1)},
    "Megabucks": {"main": (1, 49, 6)},
    "SuperCash": {"main": (1, 39, 4)},
    "Badger 5": {"main": (1, 31, 5)},
}

def _connect_sheet() -> Optional[gspread.Spreadsheet]:
    sheet_name = os.getenv("SHEET_NAME", "").strip()
    if not sheet_name or gspread is None:
        return None
    try:
        sa_info = json.loads(os.getenv("GOOGLE_CREDS_JSON", "{}"))
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc.open(sheet_name)
    except Exception as e:
        print(f"::warning::Sheets connect failed: {e}")
        return None

def with_backoff(fn, max_wait=float(os.getenv("GSPREAD_BACKOFF_MAX_SEC", "20")), tries=5):
    base = 0.8
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("429", "quota", "rate", "deadline", "503", "500", "backend error")) and i < tries - 1:
                sleep = min(max_wait, base * (2 ** i) * (1 + random.random()))
                print(f"[retry] {e} -> sleeping {sleep:.2f}s")
                time.sleep(sleep)
                continue
            raise

def fmt_pick(game: str, nums: List[int], extra: Optional[int] = None) -> str:
    if game == "Powerball":
        return f"{' '.join(map(str, sorted(nums)))} | PB {extra}"
    return " ".join(map(str, sorted(nums)))

def load_recent_set(ss: Optional[gspread.Spreadsheet], game: str, max_rows: int = 200) -> Set[str]:
    if ss is None: return set()
    title = f"{game}_Predictions"
    try:
        ws = ss.worksheet(title)
    except Exception:
        return set()
    try:
        values = with_backoff(lambda: ws.get_all_values())[-max_rows:]
        s = { row[0].strip() for row in values if row and row[0].strip() }
        print(f"[markov_mc] recent[{game}] unique={len(s)} (from last {len(values)} rows)")
        return s
    except Exception as e:
        print(f"::warning::read recent failed: {e}")
        return set()

def write_predictions(ss: Optional[gspread.Spreadsheet], game: str, picks: List[str]) -> None:
    if ss is None:
        for p in picks: print(f"[{game}] {p}")
        return
    title = f"{game}_Predictions"
    try:
        try:
            ws = ss.worksheet(title)
        except Exception:
            ws = ss.add_worksheet(title, rows=2000, cols=3)
        rows = [[p, datetime.now(timezone.utc).isoformat(), "markov_mc"] for p in picks]
        with_backoff(lambda: ws.append_rows(rows))
        print(f"[markov_mc] wrote {len(rows)} predictions for {game}.")
    except Exception as e:
        print(f"::warning::write failed: {e}")
        for p in picks: print(f"[{game}] {p}")

# ---- Placeholder "Markov" selection with temperature noise ----
def soft_sample(population: List[int], temp: float) -> int:
    # Biased but simple: weight by 1/rank with noise
    ranks = list(range(1, len(population)+1))
    weights = np.array([1.0/r for r in ranks], dtype=float)
    # add temperature noise
    noise = np.random.random(len(population)) ** (1.0 / max(1e-6, temp))
    weights = weights * (0.75 + 0.5*noise)
    probs = weights / weights.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]

def markov_generate_one_pick(game: str) -> str:
    rules = GAME_RULES[game]
    low, high, k = rules["main"]
    universe = list(range(low, high+1))
    chosen: Set[int] = set()
    while len(chosen) < k:
        n = soft_sample(universe, MARKOV_TEMP)
        if n not in chosen:
            chosen.add(n)
    extra = None
    if game == "Powerball":
        pb_low, pb_high, pb_k = rules["power"]
        extra = random.randint(pb_low, pb_high)
    return fmt_pick(game, list(chosen), extra)

def generate_markov_for_game(game: str, target: int, recent: Set[str]) -> List[str]:
    out: List[str] = []
    guard = 0
    while len(out) < target and guard < 10000:
        guard += 1
        cand = markov_generate_one_pick(game)
        if cand in recent or cand in out:
            continue
        out.append(cand)
    return out

def main():
    ss = _connect_sheet()
    targets = {}
    # honor MARKOV_TARGETS_JSON if present
    raw_targets = os.getenv("MARKOV_TARGETS_JSON", "").strip()
    if raw_targets:
        try:
            targets = json.loads(raw_targets)
        except Exception as e:
            print(f"::warning::MARKOV_TARGETS_JSON invalid: {e}")
    if not targets:
        targets = {"SuperCash": 10, "Badger 5": 5, "Megabucks": 10, "Powerball": 5}

    total = 0
    for game, count in targets.items():
        count = int(count or 0)
        if count <= 0: 
            continue
        if game not in GAME_RULES:
            print(f"::warning::Unknown game '{game}', skipping.")
            continue
        recent = load_recent_set(ss, game, max_rows=200)
        picks = generate_markov_for_game(game, count, recent)
        write_predictions(ss, game, picks)
        total += len(picks)
    print(f"[markov_mc] Finished. Emitted total={total} predictions.")

if __name__ == "__main__":
    main()
