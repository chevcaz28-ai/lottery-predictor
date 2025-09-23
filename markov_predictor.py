#!/usr/bin/env python3
"""
Markov-based predictor that reads historical draws from your Google Sheet,
builds a simple first-order Markov model over numbers, and writes predictions
into the game's Predictions sheet as a separate method named 'markov'.

It is designed to run *alongside* main.py and not import or modify it.

Required env:
- GOOGLE_CREDS_JSON        (service-account JSON)
- SHEET_NAME               (e.g. "Lottery Predictor New August 25")
- LOCAL_TZ                 (e.g. "America/Chicago")
- ENABLE_MARKOV_METHOD     ("1" to run; anything else to skip)
- MARKOV_TARGETS_JSON      (optional) JSON mapping of game->count to emit, e.g. {"SuperCash":10,"Badger 5":5}
- MARKOV_TEMP              (optional float, e.g. "0.7") softmax temperature
- MARKOV_SMOOTH            (optional float, e.g. "1.0") Laplace smoothing
- MARKOV_METHOD_NAME       (optional str, default "markov")

Assumptions about your sheet tabs (match your existing layout):
- A tab per results schema, e.g. "SuperCash_Results", "Badger 5_Results", etc.
  with a header row and newest draw on row 2 (like your current flow).
- A tab per Predictions, e.g. "SuperCash_Predictions", etc., with the same
  header structure your main.py writes (this script appends rows using a
  METHOD name "markov" to keep things separate/safe).

This script only uses public packages you already install in CI.
"""

import os, json, math, time, random
import datetime as dt
from typing import List, Dict, Any, Tuple
import pytz
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# ---------- Configuration helpers ----------

def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip() in ("1","true","True","YES","yes","on","ON")

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def now_local(tzname: str) -> dt.datetime:
    tz = pytz.timezone(tzname or "UTC")
    return dt.datetime.now(tz)

def today_local_str(tzname: str) -> str:
    return now_local(tzname).strftime("%Y-%m-%d")

# ---------- Sheets I/O ----------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
def open_sheet(sheet_name: str):
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc.open(sheet_name)

def get_or_create(ws_parent, title: str, rows: int = 5000, cols: int = 30):
    try:
        return ws_parent.worksheet(title)
    except gspread.WorksheetNotFound:
        return ws_parent.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def read_values(ws) -> List[List[str]]:
    vals = ws.get_all_values()
    # strip empty trailing columns for safety
    return [list(map(str, row)) for row in vals]

def append_rows(ws, rows: List[List[Any]]):
    if not rows: return
    ws.append_rows(rows, value_input_option="USER_ENTERED")

# ---------- Lightweight Markov model over numbers ----------

def build_markov(
    draws: List[List[int]],
    max_num: int,
    laplace: float = 1.0
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Flatten draws (ascending within a draw) into a 1-D sequence of numbers
    and build a transition matrix T[i,j] ~ P(next=j | current=i).

    Returns:
      - transition probability DataFrame shape (max_num+1, max_num+1)
      - global frequency prior (length max_num+1)
    """
    # 1-based indexing convenience; index 0 unused
    counts = pd.DataFrame(laplace, index=range(1, max_num+1), columns=range(1, max_num+1), dtype=float)

    # Global frequency prior
    prior = [0.0]*(max_num+1)

    # Create a long sequence by chaining draws; we’ll chain each draw internally
    # and also chain last of draw k to first of draw k+1 to allow inter-draw transitions.
    seq: List[int] = []
    for d in draws:
        d_sorted = sorted(int(x) for x in d if x)  # keep consistent ordering
        if not d_sorted: continue
        seq.extend(d_sorted)
    # also add cross-draw edges
    for k in range(len(draws)-1):
        a = sorted(int(x) for x in draws[k] if x)
        b = sorted(int(x) for x in draws[k+1] if x)
        if a and b:
            seq.append(a[-1])
            seq.append(b[0])

    # Count transitions
    for i in range(len(seq)-1):
        cur, nxt = seq[i], seq[i+1]
        if 1 <= cur <= max_num and 1 <= nxt <= max_num:
            counts.loc[cur, nxt] += 1.0

    # Prior
    for x in seq:
        if 1 <= x <= max_num:
            prior[x] += 1.0
    # Laplace on prior too
    prior = [p + laplace for p in prior]

    # Normalize rows to probabilities
    probs = counts.div(counts.sum(axis=1).replace(0.0, 1.0), axis=0)
    return probs, prior

def softmax(xs: List[float], temp: float) -> List[float]:
    if temp <= 1e-9:
        # practically argmax
        out = [0.0]*len(xs)
        out[int(max(range(len(xs)), key=lambda i: xs[i]))] = 1.0
        return out
    m = max(xs)
    exps = [math.exp((x - m)/temp) for x in xs]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def markov_pick(
    last_draw: List[int],
    T: pd.DataFrame,
    prior: List[float],
    n_numbers: int,
    max_num: int,
    temperature: float = 0.7
) -> List[int]:
    """
    Produce one combination by averaging transition rows from the last draw
    (plus prior), then sampling greedily without replacement via softmax.
    """
    last = sorted(int(x) for x in last_draw if x and 1 <= int(x) <= max_num)
    if not last:
        last = [i for i in range(1, min(max_num, n_numbers)+1)]

    # Average transition probs from all seeds in last draw
    avg = [0.0]*(max_num+1)
    for seed in last:
        row = T.loc[seed].values  # length max_num
        for j in range(max_num):
            avg[j+1] += row[j]
    for j in range(1, max_num+1):
        avg[j] = (avg[j] / max(1, len(last))) + (prior[j] / (sum(prior) or 1.0))*0.15  # small prior mix

    chosen = set()
    picks: List[int] = []
    for _ in range(n_numbers):
        # mask already-picked
        scores = [0.0] + [ (0.0 if j in chosen else avg[j]) for j in range(1, max_num+1) ]
        probs = softmax(scores[1:], temperature)  # drop index 0 for softmax
        # sample the argmax of probs (deterministic by design to keep CI stable)
        j_star = max(range(1, max_num+1), key=lambda j: (0.0 if j in chosen else probs[j-1]))
        chosen.add(j_star)
        picks.append(j_star)

    return sorted(picks)

# ---------- Game configuration ----------

GAME_CONFIG = {
    # name in Results sheet tab   numbers_per_draw,  max_number
    "SuperCash":       (6, 39),
    "Badger 5":        (5, 31),
    "Megabucks":       (6, 49),
    "Powerball":       (5, 69),   # main balls only; PB is separate & out-of-scope here
}

def results_tab(game: str) -> str:
    return f"{game}_Results"

def predictions_tab(game: str) -> str:
    return f"{game}_Predictions"

# ---------- Glue to Sheet ----------

def read_history(ws_results) -> List[List[int]]:
    vals = read_values(ws_results)
    if not vals: return []
    header = vals[0]
    rows = vals[1:]
    # heuristic: read the number columns by scanning header tokens that look like "N1","N2",... or plain numbers
    num_cols = []
    for i, h in enumerate(header):
        h2 = h.strip().lower()
        if h2.startswith("n") and any(ch.isdigit() for ch in h2):
            num_cols.append(i)
        elif h2.isdigit():
            num_cols.append(i)
    if not num_cols:
        # fallback: assume columns 2..7 hold numbers (common in your sheets)
        num_cols = list(range(1, 7))
    out: List[List[int]] = []
    for r in rows:
        try:
            nums = [int(x) for i,x in enumerate(r) if i in num_cols and x.strip().isdigit()]
            if nums:
                out.append(nums)
        except Exception:
            continue
    return out

def write_predictions(ws_pred, game: str, method_name: str, combos: List[List[int]], next_draw_date: str):
    """
    Appends rows with: DateEmitted, Method, NextDrawDate, and the numbers.
    We try to match a common structure; adjust if your prediction tabs differ.
    """
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    rows = []
    for nums in combos:
        row = [today, method_name, next_draw_date] + nums
        rows.append(row)
    append_rows(ws_pred, rows)

# ---------- Runner ----------

def run_for_game(ss, game: str, emit_count: int, temp: float, smooth: float, method_name: str, tzname: str):
    if game not in GAME_CONFIG:
        print(f"[markov] Skipping unknown game: {game}")
        return
    n_numbers, max_num = GAME_CONFIG[game]
    ws_res = get_or_create(ss, results_tab(game))
    hist = read_history(ws_res)
    if len(hist) < 15:  # require minimum history
        print(f"[markov] Not enough history for {game}: {len(hist)} rows.")
        return

    T, prior = build_markov(hist, max_num=max_num, laplace=smooth)
    last = hist[0]  # assuming row 2 is newest in your sheets (same as main.py)
    combos = []
    for _ in range(emit_count):
        picks = markov_pick(last, T, prior, n_numbers=n_numbers, max_num=max_num, temperature=temp)
        combos.append(picks)

    # Best effort to get next draw date: use today if uncertain (keeps writer happy)
    next_draw = today_local_str(tzname)
    ws_pred = get_or_create(ss, predictions_tab(game))
    write_predictions(ws_pred, game, method_name, combos, next_draw)
    print(f"[markov] Wrote {len(combos)} predictions for {game} using method '{method_name}'.")

def main():
    if not env_bool("ENABLE_MARKOV_METHOD", False):
        print("[markov] ENABLE_MARKOV_METHOD is not enabled; exiting.")
        return

    sheet_name = os.getenv("SHEET_NAME") or "Lottery Predictor"
    tzname = os.getenv("LOCAL_TZ", "America/Chicago")
    method_name = os.getenv("MARKOV_METHOD_NAME", "markov")
    temp = env_float("MARKOV_TEMP", 0.7)
    smooth = env_float("MARKOV_SMOOTH", 1.0)

    # Which games and how many rows to emit?
    default_targets = {
        "SuperCash": 10,
        "Badger 5": 5,
        # Add others if you want:
        # "Megabucks": 10,
        # "Powerball": 5,
    }
    try:
        targets = json.loads(os.getenv("MARKOV_TARGETS_JSON","")) or default_targets
    except Exception:
        targets = default_targets

    ss = open_sheet(sheet_name)
    for game, count in targets.items():
        try:
            run_for_game(ss, game, int(count), temp, smooth, method_name, tzname)
        except Exception as e:
            print(f"[markov] Error for {game}: {e}")

if __name__ == "__main__":
    main()
