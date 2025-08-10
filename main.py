#!/usr/bin/env python3
"""
main.py — Integrates with existing tracker sheets (results only, no ML yet)
---------------------------------------------------------------------------
This script will:
- Validate env vars and connect to your Google Sheet.
- Update ONLY these existing result tabs with the exact columns you already use:
    Powerball_Results: Date, N1..N5, PB
    Megabucks_Results: Date, N1..N6
    SuperCash_Results: Date, N1..N6, Doubler
    Badger5_Results:   Date, N1..N5
- Append a timestamp to Health_Check and keep Raw_API_WI + Games_Index for debugging.
- (Predictions are NOT written yet—after you confirm results load correctly, we’ll add that.)

Safe behavior:
- Each results tab is cleared and fully rewritten from the API’s current history.
- If a game’s draws can’t be parsed, the tab will contain a single row noting the parser issue.
"""

import os
import json
import datetime as dt
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
import gspread
from google.oauth2.service_account import Credentials

# -------------------------
# Config
# -------------------------
DEFAULT_SHEET_NAME = "Lottery Predictor New August 25"
API_URL_WI = "https://lottery-results.p.rapidapi.com/games-by-state/us/wi"
API_HOST = "lottery-results.p.rapidapi.com"

RAW_MAX_ROWS = 5000

# Expected column headers by tab
TABS_SCHEMA = {
    "Powerball_Results": ["Date", "N1", "N2", "N3", "N4", "N5", "PB"],
    "Megabucks_Results": ["Date", "N1", "N2", "N3", "N4", "N5", "N6"],
    "SuperCash_Results": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Doubler"],
    "Badger5_Results":   ["Date", "N1", "N2", "N3", "N4", "N5"],
}

# -------------------------
# Helpers
# -------------------------
def fail(msg: str) -> None:
    raise RuntimeError(msg)

def load_runtime_config():
    sheet_name = os.getenv("SHEET_NAME", DEFAULT_SHEET_NAME)
    rapid_key = os.getenv("RAPID_API_KEY")
    google_creds_json = os.getenv("GOOGLE_CREDS_JSON")

    missing = []
    if not rapid_key:
        missing.append("RAPID_API_KEY")
    if not google_creds_json:
        missing.append("GOOGLE_CREDS_JSON")
    if missing:
        fail("Missing required environment variables: " + ", ".join(missing))

    try:
        creds_dict = json.loads(google_creds_json)
    except json.JSONDecodeError:
        fail("GOOGLE_CREDS_JSON is not valid JSON. Paste the FULL service account JSON.")

    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
    )
    client = gspread.authorize(creds)
    return sheet_name, rapid_key, client, creds_dict.get("client_email")

def open_sheet(client: gspread.Client, sheet_name: str) -> gspread.Spreadsheet:
    try:
        return client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        fail(f'Google Sheet "{sheet_name}" was not found or not shared with the service account.')

def get_or_create_worksheet(ss: gspread.Spreadsheet, title: str, rows: int = 1000, cols: int = 26) -> gspread.Worksheet:
    try:
        return ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def append_health_check(ws_health: gspread.Worksheet) -> None:
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    headers = ["Timestamp_UTC", "Status"]
    cur = ws_health.get_all_values()
    if not cur or (cur and cur[0] != headers):
        ws_health.clear()
        ws_health.append_row(headers, value_input_option="RAW")
    ws_health.append_row([now_utc, "OK"], value_input_option="RAW")

def fetch_wi_results(rapid_key: str) -> Dict[str, Any]:
    headers = {"x-rapidapi-key": rapid_key, "x-rapidapi-host": API_HOST}
    r = requests.get(API_URL_WI, headers=headers, timeout=45)
    r.raise_for_status()
    return r.json()

def write_raw_api(ws_raw: gspread.Worksheet, data: Dict[str, Any]) -> None:
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    lines = pretty.splitlines()
    if len(lines) > RAW_MAX_ROWS:
        lines = lines[-RAW_MAX_ROWS:]
    ws_raw.clear()
    ws_raw.update("A1", [["Raw JSON (pretty)"]])
    ws_raw.update("A2", [[line] for line in lines] or [["<empty>"]], value_input_option="RAW")

def detect_games_list(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("data"), list):
        return [x for x in data["data"] if isinstance(x, dict)]
    if isinstance(data.get("games"), list):
        return [x for x in data["games"] if isinstance(x, dict)]
    # nested
    for v in data.values():
        if isinstance(v, dict):
            if isinstance(v.get("data"), list):
                return [x for x in v["data"] if isinstance(x, dict)]
            if isinstance(v.get("games"), list):
                return [x for x in v["games"] if isinstance(x, dict)]
    return None

def as_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""

def normalize_date(value: Any) -> str:
    s = as_str(value).strip()
    if not s:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
    return s

def to_int_list(v: Any) -> List[Optional[int]]:
    """Turn a list or a comma/space string into a list of ints (non-digits become None)."""
    if isinstance(v, list):
        parts = v
    else:
        s = as_str(v)
        parts = re.split(r"[,\s]+", s.strip()) if s else []
    out: List[Optional[int]] = []
    for p in parts:
        try:
            out.append(int(str(p).strip()))
        except Exception:
            if str(p).strip():
                out.append(None)
    return out

def pick_first(keys: List[str], d: Dict[str, Any]) -> Any:
    low = {k.lower(): k for k in d.keys()}
    for k in keys:
        if k.lower() in low:
            return d[low[k.lower()]]
    return None

def extract_draws_generic(game: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of draw dicts with at least date + a numbers container + possible extras."""
    draws = None
    for key in ["draws", "results", "latest_results", "past_results", "history", "recent_draws"]:
        v = game.get(key)
        if isinstance(v, list):
            draws = v
            break
    if draws is None and isinstance(game.get("latest_result"), dict):
        draws = [game["latest_result"]]
    return [d for d in (draws or []) if isinstance(d, dict)]

def parse_powerball_rows(draws: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for d in draws:
        date = normalize_date(pick_first(["date","draw_date","time","timestamp"], d))
        # common shapes
        nums = pick_first(["numbers","winning_numbers","winningNumbers","balls","regular","main"], d)
        pb = pick_first(["powerball","pb","bonus_ball","bonusBall"], d)
        # try nested
        if nums is None:
            wn = pick_first(["winningNumbers","winning_numbers"], d)
            if isinstance(wn, dict):
                nums = wn.get("regular") or wn.get("main")
                pb = pb or wn.get("bonus") or wn.get("powerball")
        arr = [x for x in to_int_list(nums) if x is not None]
        n1,n2,n3,n4,n5 = (arr + [None]*5)[:5]
        pbv = None
        if pb is not None:
            lst = [x for x in to_int_list(pb) if x is not None]
            pbv = lst[0] if lst else None
        rows.append([date, n1,n2,n3,n4,n5, pbv])
    # newest first
    rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    return rows

def parse_megabucks_rows(draws: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for d in draws:
        date = normalize_date(pick_first(["date","draw_date","time","timestamp"], d))
        nums = pick_first(["numbers","winning_numbers","winningNumbers","balls","regular","main","primary"], d)
        if nums is None:
            wn = pick_first(["winningNumbers","winning_numbers"], d)
            if isinstance(wn, dict):
                nums = wn.get("regular") or wn.get("main") or wn.get("primary")
        arr = [x for x in to_int_list(nums) if x is not None]
        n1,n2,n3,n4,n5,n6 = (arr + [None]*6)[:6]
        rows.append([date, n1,n2,n3,n4,n5,n6])
    rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    return rows

def parse_supercash_rows(draws: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for d in draws:
        date = normalize_date(pick_first(["date","draw_date","time","timestamp"], d))
        nums = pick_first(["numbers","winning_numbers","winningNumbers","balls","regular","main","primary"], d)
        doubler = pick_first(["doubler","doubler_flag","isDoubler","double"], d)
        if nums is None:
            wn = pick_first(["winningNumbers","winning_numbers"], d)
            if isinstance(wn, dict):
                nums = wn.get("regular") or wn.get("main") or wn.get("primary")
                doubler = doubler or wn.get("doubler")
        arr = [x for x in to_int_list(nums) if x is not None]
        n1,n2,n3,n4,n5,n6 = (arr + [None]*6)[:6]
        # Normalize doubler to Yes/No/blank
        dstr = as_str(doubler).strip().lower()
        if dstr in ["true","yes","y","1"]:
            dnorm = "Yes"
        elif dstr in ["false","no","n","0",""]:
            dnorm = "No" if dstr != "" else ""
        else:
            dnorm = as_str(doubler) if dstr else ""
        rows.append([date, n1,n2,n3,n4,n5,n6, dnorm])
    rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    return rows

def parse_badger5_rows(draws: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for d in draws:
        date = normalize_date(pick_first(["date","draw_date","time","timestamp"], d))
        nums = pick_first(["numbers","winning_numbers","winningNumbers","balls","regular","main","primary"], d)
        if nums is None:
            wn = pick_first(["winningNumbers","winning_numbers"], d)
            if isinstance(wn, dict):
                nums = wn.get("regular") or wn.get("main") or wn.get("primary")
        arr = [x for x in to_int_list(nums) if x is not None]
        n1,n2,n3,n4,n5 = (arr + [None]*5)[:5]
        rows.append([date, n1,n2,n3,n4,n5])
    rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    return rows

def write_rows(ws, headers: List[str], rows: List[List[Any]], game_label: str):
    ws.clear()
    ws.update("A1", [headers])
    if rows:
        ws.update("A2", rows, value_input_option="RAW")
    else:
        ws.update("A2", [["", "", "", f"No draws found for {game_label}"]])

def write_games_index(ws_index: gspread.Worksheet, games: List[Dict[str, Any]]) -> None:
    headers = ["#", "game_id", "game_name"]
    rows: List[List[str]] = []
    for i, g in enumerate(games, start=1):
        gid = as_str(g.get("id", ""))
        name = as_str(g.get("name", ""))
        rows.append([i, gid, name])
    ws_index.clear()
    ws_index.update("A1", [headers])
    ws_index.update("A2", rows or [["", "", "No games found"]], value_input_option="RAW")

# -------------------------
# Main
# -------------------------
def main():
    sheet_name, rapid_key, client, sa_email = load_runtime_config()
    ss = open_sheet(client, sheet_name)

    ws_health = get_or_create_worksheet(ss, "Health_Check", rows=100, cols=3)
    ws_raw = get_or_create_worksheet(ss, "Raw_API_WI", rows=RAW_MAX_ROWS + 10, cols=1)
    ws_index = get_or_create_worksheet(ss, "Games_Index", rows=200, cols=3)

    append_health_check(ws_health)

    data = fetch_wi_results(rapid_key)
    write_raw_api(ws_raw, data)

    games_list = detect_games_list(data) or []
    write_games_index(ws_index, games_list)

    # Build a quick lookup by name/id
    def game_contains(g: Dict[str, Any], needle: str) -> bool:
        s = (as_str(g.get("name")) + " " + as_str(g.get("id")) + " " + as_str(g.get("title"))).lower()
        return needle.lower() in s

    # Parse and write each tab
    targets = [
        ("Powerball_Results", parse_powerball_rows, "powerball"),
        ("Megabucks_Results", parse_megabucks_rows, "megabucks"),
        ("SuperCash_Results", parse_supercash_rows, "super cash"),
        ("Badger5_Results",   parse_badger5_rows, "badger 5"),
    ]

    for tab_name, parser, key in targets:
        ws = get_or_create_worksheet(ss, tab_name, rows=2000, cols=len(TABS_SCHEMA[tab_name]))
        headers = TABS_SCHEMA[tab_name]

        # find the matching game in the API list
        matched = None
        for g in games_list:
            if game_contains(g, key):
                matched = g
                break

        if not matched:
            write_rows(ws, headers, [], tab_name.replace("_Results",""))
            continue

        draws = extract_draws_generic(matched)
        try:
            rows = parser(draws)
        except Exception as e:
            rows = [[ "", "", "", f"Parser error: {e}" ]]
        write_rows(ws, headers, rows, tab_name.replace("_Results",""))

    # Console summary
    print("Success! Results updated in existing tabs:")
    for t in TABS_SCHEMA.keys():
        print(f" - {t}")
    if sa_email:
        print(f"(Ensure the sheet is shared with: {sa_email})")

if __name__ == "__main__":
    main()
