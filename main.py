#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import gspread
from google.oauth2.service_account import Credentials

def get_args():
    p = argparse.ArgumentParser(description="Lottery predictions with idempotency and exact counts.")
    p.add_argument("--force", action="store_true", help="Generate even if no new draws are detected.")
    p.add_argument("--dry-run", action="store_true", help="Run logic without writing predictions or updating state.")
    p.add_argument("--only-game", type=str, default="", help="If provided, limit to a single game name (e.g., 'Badger 5').")
    return p.parse_args()

SHEET_NAME = os.getenv("SHEET_NAME", "Lottery Predictor New August 25")

TARGET_COUNTS = {
    "Powerball": int(os.getenv("PRED_COUNT_POWERBALL", "20")),
    "Megabucks": int(os.getenv("PRED_COUNT_MEGABUCKS", "20")),
    "Super Cash": int(os.getenv("PRED_COUNT_SUPERCASH", "20")),
    "Badger 5": int(os.getenv("PRED_COUNT_BADGER5", "20")),
}

ENABLE_PREDICTIONS = os.getenv("ENABLE_PREDICTIONS", "1").lower() in {"1","true","yes"}

RESULT_TABS = {
    "Powerball": "Powerball_Results",
    "Megabucks": "Megabucks_Results",
    "Super Cash": "SuperCash_Results",
    "Badger 5": "Badger5_Results",
}
PREDICTIONS_TAB = "Prediction_Tracker"
STATE_TAB = "State"
RUN_LOG_TAB = "Run_Log"

def open_sheet() -> gspread.Spreadsheet:
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open(SHEET_NAME)

def get_or_create_ws(ss: gspread.Spreadsheet, title: str, headers: List[str]) -> gspread.Worksheet:
    try:
        ws = ss.worksheet(title)
        values = ws.get_all_values()
        if not values:
            ws.append_row(headers, value_input_option="RAW")
        return ws
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=title, rows="1000", cols=str(max(10, len(headers)+5)))
        ws.append_row(headers, value_input_option="RAW")
        return ws

STATE_HEADERS = ["Game", "LastProcessedKey", "UpdatedAtUTC"]

def read_state_map(state_ws) -> Dict[str, str]:
    data = state_ws.get_all_values()
    if not data:
        return {}
    header = {name: idx for idx, name in enumerate(data[0])}
    out: Dict[str,str] = {}
    for row in data[1:]:
        if not row or len(row) <= header["Game"]:
            continue
        game = (row[header["Game"]] or "").strip()
        key  = (row[header["LastProcessedKey"]] if len(row) > header["LastProcessedKey"] else "").strip()
        if game:
            out[game] = key
    return out

def upsert_state_key(state_ws, game: str, batch_key: str):
    data = state_ws.get_all_values() or [STATE_HEADERS]
    header = {name: idx for idx, name in enumerate(data[0])}
    row_idx = None
    for i, row in enumerate(data[1:], start=2):
        if row and len(row) > header["Game"] and (row[header["Game"]] or "").strip() == game:
            row_idx = i
            break
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if row_idx is None:
        state_ws.append_row([game, batch_key, now_utc], value_input_option="RAW")
    else:
        state_ws.update_cell(row_idx, header["LastProcessedKey"]+1, batch_key)
        state_ws.update_cell(row_idx, header["UpdatedAtUTC"]+1, now_utc)

def parse_results(ws: gspread.Worksheet, needs_bonus: bool=False, needs_six: bool=False) -> List[Dict[str,Any]]:
    data = ws.get_all_values()
    if not data or not data[0]:
        return []
    hdr = {name: idx for idx, name in enumerate(data[0])}
    rows: List[Dict[str,Any]] = []
    for r in data[1:]:
        if not r:
            continue
        date_str = (r[hdr.get("Date", 0)] if len(r) > hdr.get("Date", 0) else "").strip()
        if not date_str:
            continue
        try:
            d = datetime.fromisoformat(date_str.replace("Z","").split(" ")[0])
        except Exception:
            try:
                d = datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
            except Exception:
                continue
        date_iso = d.strftime("%Y-%m-%d")

        nums = []
        for key in ["N1","N2","N3","N4","N5"]:
            i = hdr.get(key)
            if i is not None and len(r) > i and (r[i] or "").strip():
                try:
                    nums.append(int(str(r[i]).split(":")[-1]))
                except:
                    pass
        if needs_six:
            i = hdr.get("N6")
            if i is not None and len(r) > i and (r[i] or "").strip():
                try:
                    nums.append(int(str(r[i]).split(":")[-1]))
                except:
                    pass

        bonus = None
        if needs_bonus:
            i = hdr.get("PB")
            if i is not None and len(r) > i and (r[i] or "").strip():
                try:
                    bonus = int(str(r[i]).split(":")[-1])
                except:
                    bonus = None

        rows.append({"Date": date_iso, "nums": nums, "bonus": bonus})
    rows.sort(key=lambda x: x["Date"])
    return rows

def latest_draw_info(ss: gspread.Spreadsheet, game: str) -> Optional[Dict[str,Any]]:
    tab = RESULT_TABS[game]
    try:
        ws = ss.worksheet(tab)
    except gspread.WorksheetNotFound:
        return None
    needs_bonus = (game == "Powerball")
    needs_six   = (game in {"Megabucks","Super Cash"})
    rows = parse_results(ws, needs_bonus=needs_bonus, needs_six=needs_six)
    if not rows:
        return None
    latest = rows[-1]
    recent = [r for r in rows[-12:]][::-1]
    return {
        "draw_date": latest["Date"],
        "recent_draws": [r["nums"] + (([r["bonus"]] if r["bonus"] is not None else [])) for r in recent],
    }

PRED_HEADERS = ["Timestamp","Game","Prediction","Method","Win Count","Matches","Match Count","BatchKey"]

def ensure_pred_headers(ws: gspread.Worksheet):
    values = ws.get_all_values()
    if not values:
        ws.append_row(PRED_HEADERS, value_input_option="RAW")
        return
    header = values[0]
    if "BatchKey" not in header:
        ws.update_cell(1, len(header)+1, "BatchKey")

def predictions_exist(ws: gspread.Worksheet, game: str, batch_key: str) -> bool:
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return False
    header = {name: idx for idx, name in enumerate(values[0])}
    gi = header.get("Game"); bi = header.get("BatchKey")
    if gi is None or bi is None:
        return False
    for row in values[1:]:
        if not row or len(row) <= max(gi, bi):
            continue
        if row[gi] == game and row[bi] == batch_key:
            return True
    return False

def append_predictions(ws: gspread.Worksheet, rows: List[List[Any]]):
    if not rows:
        return
    ws.append_rows(rows, value_input_option="RAW")

from random import sample, randint

GAME_CFG = {
    "Powerball": {"main_count": 5, "main_min": 1, "main_max": 69, "bonus_min": 1, "bonus_max": 26},
    "Megabucks": {"main_count": 6, "main_min": 1, "main_max": 49},
    "Super Cash": {"main_count": 6, "main_min": 1, "main_max": 39},
    "Badger 5": {"main_count": 5, "main_min": 1, "main_max": 31},
}

def draw_candidate(game: str) -> List[int]:
    cfg = GAME_CFG[game]
    mains = sorted(sample(range(cfg["main_min"], cfg["main_max"]+1), cfg["main_count"]))
    if game == "Powerball":
        pb = randint(cfg["bonus_min"], cfg["bonus_max"])
        return mains + [pb]
    return mains

def passes_constraints(game: str, pick: List[int], strict_level: int, recent: List[List[int]]) -> bool:
    base_depth = 8
    depth = max(0, base_depth - 2*strict_level)
    for past in recent[:depth]:
        if pick == past:
            return False
    s = sum(pick[:-1]) if game == "Powerball" else sum(pick)
    base_lo, base_hi = 60, 220
    wiggle = 12 * strict_level
    if not (base_lo - wiggle <= s <= base_hi + wiggle):
        return False
    return True

def generate_batch(game: str, target: int, recent: List[List[int]]) -> Tuple[List[List[int]], int]:
    picks: List[List[int]] = []
    seen = set()
    strict = 0
    MAX_STRICT = 6
    MAX_ATTEMPTS = 5000
    attempts = 0
    while len(picks) < target and attempts < MAX_ATTEMPTS:
        attempts += 1
        c = draw_candidate(game)
        t = tuple(c)
        if t in seen:
            continue
        if passes_constraints(game, c, strict, recent):
            picks.append(c); seen.add(t); continue
        if attempts % 1000 == 0 and strict < MAX_STRICT:
            strict += 1
    while len(picks) < target:
        c = draw_candidate(game)
        picks.append(c)
    return picks, strict

def format_prediction(game: str, pick: List[int]) -> str:
    if game == "Powerball":
        mains = pick[:-1]; pb = pick[-1]
        return "{} PB:{}".format("-".join(map(str, mains)), pb)
    return "-".join(map(str, pick))

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def run():
    args = get_args()
    if not ENABLE_PREDICTIONS and not args.force:
        print("ENABLE_PREDICTIONS is false; exiting.")
        return

    ss = open_sheet()
    state_ws = get_or_create_ws(ss, STATE_TAB, STATE_HEADERS)
    pred_ws  = get_or_create_ws(ss, PREDICTIONS_TAB, PRED_HEADERS)
    ensure_pred_headers(pred_ws)

    state_map = read_state_map(state_ws)
    games = list(RESULT_TABS.keys())
    if args.only_game:
        games = [g for g in games if g.lower() == args.only_game.lower()]
        if not games:
            print(f"No game matched '--only-game {args.only_game}'."); return

    for game in games:
        target = TARGET_COUNTS.get(game, 20)
        info = latest_draw_info(ss, game)
        if not info:
            print(f"[{game}] No results found; skipping."); continue
        draw_date = info["draw_date"]
        recent    = info["recent_draws"]
        batch_key = draw_date

        last_key = state_map.get(game, "")
        is_new = (batch_key != last_key)
        print(f"[{game}] latest={batch_key} last={last_key or '—'} is_new={is_new} force={args.force}")

        if not is_new and not args.force:
            print(f"[{game}] No new draw and --force not set; skipping."); continue

        if predictions_exist(pred_ws, game, batch_key) and not args.force:
            print(f"[{game}] Predictions already exist for BatchKey {batch_key}; skipping.")
            upsert_state_key(state_ws, game, batch_key); continue

        picks, strict_used = generate_batch(game, target, recent)
        ts = now_utc_str()
        rows = [[ts, game, format_prediction(game, p), "auto_v4", 0, "", 0, batch_key] for p in picks]
        print(f"[{game}] Writing {len(rows)} predictions for batch {batch_key} (relax_level={strict_used})")
        if not args.dry_run:
            append_predictions(pred_ws, rows)
            upsert_state_key(state_ws, game, batch_key)
        try:
            runlog_ws = ss.worksheet(RUN_LOG_TAB)
            values = runlog_ws.get_all_values() or [["Game","LastResultDate","LastPredictedNextDraw","Notes"]]
            header = {name: idx for idx, name in enumerate(values[0])}
            row_idx = None
            for i, r in enumerate(values[1:], start=2):
                if r and len(r) > header.get("Game",0) and r[header["Game"]] == game:
                    row_idx = i; break
            if row_idx is None:
                runlog_ws.append_row([game, draw_date, "", ""], value_input_option="RAW")
            else:
                runlog_ws.update_cell(row_idx, header["LastResultDate"]+1, draw_date)
        except Exception as e:
            print(f"[{game}] Note: could not update Run_Log ({e})")
    print("Success! Finished.")

if __name__ == "__main__":
    run()
