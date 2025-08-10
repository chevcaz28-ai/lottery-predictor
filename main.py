#!/usr/bin/env python3
"""
main.py — v3.1: Results into existing tabs, protection-safe writes
------------------------------------------------------------------
Changes vs v3:
- Avoids ws.clear() so protected tabs don't error.
- Writes headers with try/except; if header row is protected we skip it.
- Overwrites the body ("A2:...") explicitly; also writes blank rows to clean leftovers.
- Uses gspread .update(values=..., range_name=...) to silence deprecation warnings.
"""

import os
import json
import datetime as dt
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError

DEFAULT_SHEET_NAME = "Lottery Predictor New August 25"
API_URL_WI = "https://lottery-results.p.rapidapi.com/games-by-state/us/wi"
API_HOST = "lottery-results.p.rapidapi.com"

RAW_MAX_ROWS = 5000

TABS_SCHEMA = {
    "Powerball_Results": ["Date", "N1", "N2", "N3", "N4", "N5", "PB"],
    "Megabucks_Results": ["Date", "N1", "N2", "N3", "N4", "N5", "N6"],
    "SuperCash_Results": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Doubler"],
    "Badger5_Results":   ["Date", "N1", "N2", "N3", "N4", "N5"],
}

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
    try:
        if not cur or (cur and cur[0] != headers):
            ws_health.update(values=[headers], range_name="A1")
        ws_health.append_row([now_utc, "OK"], value_input_option="RAW")
    except APIError as e:
        # If even append is protected, surface a clear message
        fail(f'Cannot write to "Health_Check" due to sheet protection. Add the service account as an editor for that sheet. Details: {e}')

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
    try:
        ws_raw.update(values=[["Raw JSON (pretty)"]], range_name="A1")
    except APIError:
        pass  # if header protected, skip
    # Write body starting at A2
    body = [[line] for line in lines] or [["<empty>"]]
    ws_raw.update(values=body, range_name="A2")

def detect_games_list(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(data, dict):
        return None
    for key in ("data", "games"):
        if isinstance(data.get(key), list):
            return [x for x in data[key] if isinstance(x, dict)]
    for v in data.values():
        if isinstance(v, dict):
            for key in ("data", "games"):
                if isinstance(v.get(key), list):
                    return [x for x in v[key] if isinstance(x, dict)]
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
        nums = pick_first(["numbers","winning_numbers","winningNumbers","balls","regular","main"], d)
        pb = pick_first(["powerball","pb","bonus_ball","bonusBall"], d)
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

def col_letter(n: int) -> str:
    """1 -> A, 2 -> B ..."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s or "A"

def write_rows(ws, headers: List[str], rows: List[List[Any]], game_label: str):
    # Try to write headers; if protected, skip
    try:
        ws.update(values=[headers], range_name="A1")
    except APIError:
        pass  # header row likely protected

    # Determine write range for body
    ncols = len(headers)
    max_rows = max(len(rows), 0)
    end_col = col_letter(ncols)

    # First: write the actual data
    if rows:
        ws.update(values=rows, range_name=f"A2:{end_col}{max_rows+1}")
    else:
        # Write a minimal note in the first row of the body
        note = [[ "", "", "", f"No draws found for {game_label}"]]  # fits tabs with >=4 cols
        ws.update(values=note, range_name="A2")

    # Second: attempt to "blank out" any leftover old rows within a safe window
    # (We blank up to +200 extra rows to clean previous content without clearing protected ranges.)
    BLANK_PAD = 200
    blank_rows = [["" for _ in range(ncols)] for __ in range(BLANK_PAD)]
    try:
        ws.update(values=blank_rows, range_name=f"A{max_rows+2}:{end_col}{max_rows+1+BLANK_PAD}")
    except APIError:
        # If some of those cells are protected, ignore
        pass

def write_games_index(ws_index: gspread.Worksheet, games: List[Dict[str, Any]]) -> None:
    headers = ["#", "game_id", "game_name"]
    rows: List[List[str]] = []
    for i, g in enumerate(games, start=1):
        gid = as_str(g.get("id", ""))
        name = as_str(g.get("name", ""))
        rows.append([i, gid, name])
    try:
        ws_index.update(values=[headers], range_name="A1")
    except APIError:
        pass
    ws_index.update(values=(rows or [["", "", "No games found"]]), range_name="A2")

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

    def game_contains(g: Dict[str, Any], needle: str) -> bool:
        s = (as_str(g.get("name")) + " " + as_str(g.get("id")) + " " + as_str(g.get("title"))).lower()
        return needle.lower() in s

    targets = [
        ("Powerball_Results", parse_powerball_rows, "powerball"),
        ("Megabucks_Results", parse_megabucks_rows, "megabucks"),
        ("SuperCash_Results", parse_supercash_rows, "super cash"),
        ("Badger5_Results",   parse_badger5_rows, "badger 5"),
    ]

    for tab_name, parser, key in targets:
        ws = get_or_create_worksheet(ss, tab_name, rows=2000, cols=len(TABS_SCHEMA[tab_name]))
        headers = TABS_SCHEMA[tab_name]

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

    print("Success! Results updated in existing tabs (protection-safe).")
    if sa_email:
        print(f"(If nothing changed, check range protection: add editor permission for {sa_email}.)")

if __name__ == "__main__":
    main()
