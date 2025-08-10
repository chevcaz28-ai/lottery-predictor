#!/usr/bin/env python3
"""
main.py — v3.2: Supports numeric-keyed top-level JSON with plays/draws/numbers
-----------------------------------------------------------------------------
- Reads the API shape you pasted (top-level dict with "0","1",...).
- Drills into each game's `plays` -> each play's `draws` -> `numbers` (objects with value/specialBall).
- Updates your existing four *Results tabs, protection-safe (no ws.clear()).
"""

import os, json, datetime as dt, re
from typing import Any, Dict, List, Optional, Tuple
import requests, gspread
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

def fail(msg: str): raise RuntimeError(msg)

def load_runtime_config():
    sheet_name = os.getenv("SHEET_NAME", DEFAULT_SHEET_NAME)
    rapid_key  = os.getenv("RAPID_API_KEY")
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    miss = [k for k,v in {"RAPID_API_KEY":rapid_key, "GOOGLE_CREDS_JSON":creds_json}.items() if not v]
    if miss: fail("Missing required environment variables: " + ", ".join(miss))
    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError:
        fail("GOOGLE_CREDS_JSON is not valid JSON.")
    creds = Credentials.from_service_account_info(creds_dict, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    client = gspread.authorize(creds)
    return sheet_name, rapid_key, client, creds_dict.get("client_email")

def open_sheet(client, sheet_name):
    try: return client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        fail(f'Google Sheet "{sheet_name}" not found or not shared with service account.')

def get_or_create_worksheet(ss, title, rows=1000, cols=26):
    try: return ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def append_health_check(ws_health):
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    headers = ["Timestamp_UTC","Status"]
    try: ws_health.update(values=[headers], range_name="A1")
    except APIError: pass
    ws_health.append_row([now,"OK"], value_input_option="RAW")

def fetch_wi_results(rapid_key:str)->Dict[str,Any]:
    h = {"x-rapidapi-key": rapid_key, "x-rapidapi-host": API_HOST}
    r = requests.get(API_URL_WI, headers=h, timeout=45); r.raise_for_status()
    return r.json()

def write_raw(ws_raw, data:Dict[str,Any]):
    pretty = json.dumps(data, indent=2, ensure_ascii=False).splitlines()
    try: ws_raw.update(values=[["Raw JSON (pretty)"]], range_name="A1")
    except APIError: pass
    ws_raw.update(values=[[ln] for ln in pretty], range_name="A2")

def normalize_date(s: Any) -> str:
    s = "" if s is None else str(s).strip()
    if not s: return ""
    # "08/09/2025" -> 2025-08-09
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m: mm,dd,yy = m.groups(); return f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}"
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m: return m.group(1)
    if re.match(r"^\d{4}-\d{2}-\d{2}", s): return s[:10]
    return s

def to_int(v)->Optional[int]:
    try: return int(str(v).strip())
    except: return None

def parse_numbers_objects(nums: List[Dict[str,Any]])->Tuple[List[int], Optional[int], Optional[str]]:
    """Return (mains, special_ball_number, doubler_flag) where:
       - mains are regular numbers in order
       - special ball = number attached to specialBall.name in {'Powerball','Mega Ball'} (if present)
       - doubler is 'Y'/'N' if a 'Doubler' entry is present with value 'Y'/'N'/etc.
    """
    mains: List[int] = []
    special: Optional[int] = None
    doubler: Optional[str] = None
    for item in (nums or []):
        if not isinstance(item, dict): continue
        val = item.get("value")
        sb  = item.get("specialBall")
        if sb and isinstance(sb, dict):
            name = (sb.get("name") or "").lower()
            if "powerball" in name or "mega ball" in name:
                special = to_int(val)
            elif "doubler" in name:
                v = (str(val) if val is not None else "").strip().upper()
                doubler = "Y" if v in ("Y","YES","TRUE","1") else ("N" if v in ("N","NO","FALSE","0") else v)
            else:
                # Other specials (e.g., Power Play multiplier) are ignored for core results
                pass
        else:
            iv = to_int(val)
            if iv is not None: mains.append(iv)
    return mains, special, doubler

def collect_draws_from_game(game: Dict[str,Any]) -> List[Dict[str,Any]]:
    """Game -> plays -> draws. Flatten all plays' draws (usually one play per game)."""
    out: List[Dict[str,Any]] = []
    plays = game.get("plays") if isinstance(game, dict) else None
    if not isinstance(plays, list): return out
    for p in plays:
        if not isinstance(p, dict): continue
        draws = p.get("draws")
        if not isinstance(draws, list): continue
        for d in draws:
            if not isinstance(d, dict): continue
            out.append(d)
    return out

def top_level_games(data: Dict[str,Any]) -> List[Dict[str,Any]]:
    """The API uses numeric keys "0","1",... Map them to a list of game dicts with 'name'."""
    if not isinstance(data, dict): return []
    # Try typical keys first
    if isinstance(data.get("data"), list): return [g for g in data["data"] if isinstance(g, dict)]
    if isinstance(data.get("games"), list): return [g for g in data["games"] if isinstance(g, dict)]
    # Fallback: numeric-keyed map
    games = []
    for k,v in data.items():
        if k.isdigit() and isinstance(v, dict):
            games.append(v)
    return games

def col_letter(n:int)->str:
    s=""
    while n>0:
        n, r = divmod(n-1,26)
        s = chr(65+r)+s
    return s or "A"

def write_rows(ws, headers: List[str], rows: List[List[Any]], note_label:str):
    try: ws.update(values=[headers], range_name="A1")
    except APIError: pass
    ncols = len(headers)
    end_col = col_letter(ncols)
    if rows:
        ws.update(values=rows, range_name=f"A2:{end_col}{len(rows)+1}")
    else:
        ws.update(values=[[ "", "", "", f"No draws found for {note_label}"]], range_name="A2")
    # pad blank rows to wipe leftovers (protection-safe)
    BLANK_PAD = 200
    blanks = [["" for _ in range(ncols)] for __ in range(BLANK_PAD)]
    try:
        ws.update(values=blanks, range_name=f"A{len(rows)+2}:{end_col}{len(rows)+1+BLANK_PAD}")
    except APIError:
        pass

def main():
    sheet_name, rapid_key, client, sa_email = load_runtime_config()
    ss = open_sheet(client, sheet_name)

    # Base sheets
    ws_health = get_or_create_worksheet(ss, "Health_Check", rows=100, cols=3)
    ws_raw    = get_or_create_worksheet(ss, "Raw_API_WI", rows=RAW_MAX_ROWS+10, cols=1)
    ws_index  = get_or_create_worksheet(ss, "Games_Index", rows=100, cols=3)

    append_health_check(ws_health)

    data = fetch_wi_results(rapid_key)
    write_raw(ws_raw, data)

    games = top_level_games(data)
    # Index
    idx_rows = [[i+1, str(g.get("code","")), str(g.get("name",""))] for i,g in enumerate(games)]
    try: ws_index.update(values=[["#", "game_code", "game_name"]], range_name="A1")
    except APIError: pass
    ws_index.update(values=(idx_rows or [["", "", "No games"]]), range_name="A2")

    # Helper to find a game by name contains
    def find_game(needle:str)->Optional[Dict[str,Any]]:
        n = needle.lower()
        for g in games:
            name = (str(g.get("name","")) + " " + str(g.get("code",""))).lower()
            if n in name: return g
        return None

    # Extract rows per game
    # POWERBALL
    pb_ws = get_or_create_worksheet(ss, "Powerball_Results", rows=2000, cols=len(TABS_SCHEMA["Powerball_Results"]))
    pb_game = find_game("powerball")
    pb_rows: List[List[Any]] = []
    if pb_game:
        for d in collect_draws_from_game(pb_game):
            date = normalize_date(d.get("date"))
            mains, special, _ = parse_numbers_objects(d.get("numbers", []))
            # We expect 5 mains + 1 powerball
            n1,n2,n3,n4,n5 = (mains + [None]*5)[:5]
            pb_rows.append([date, n1,n2,n3,n4,n5, special])
        pb_rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    write_rows(pb_ws, TABS_SCHEMA["Powerball_Results"], pb_rows, "Powerball")

    # MEGABUCKS
    mg_ws = get_or_create_worksheet(ss, "Megabucks_Results", rows=2000, cols=len(TABS_SCHEMA["Megabucks_Results"]))
    mg_game = find_game("megabucks")
    mg_rows: List[List[Any]] = []
    if mg_game:
        for d in collect_draws_from_game(mg_game):
            date = normalize_date(d.get("date"))
            mains, _, _ = parse_numbers_objects(d.get("numbers", []))
            n1,n2,n3,n4,n5,n6 = (mains + [None]*6)[:6]
            mg_rows.append([date, n1,n2,n3,n4,n5,n6])
        mg_rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    write_rows(mg_ws, TABS_SCHEMA["Megabucks_Results"], mg_rows, "Megabucks")

    # SUPERCASH
    sc_ws = get_or_create_worksheet(ss, "SuperCash_Results", rows=2000, cols=len(TABS_SCHEMA["SuperCash_Results"]))
    sc_game = find_game("super cash")
    sc_rows: List[List[Any]] = []
    if sc_game:
        for d in collect_draws_from_game(sc_game):
            date = normalize_date(d.get("date"))
            mains, _, doubler = parse_numbers_objects(d.get("numbers", []))
            n1,n2,n3,n4,n5,n6 = (mains + [None]*6)[:6]
            sc_rows.append([date, n1,n2,n3,n4,n5,n6, (doubler or "")])
        sc_rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    write_rows(sc_ws, TABS_SCHEMA["SuperCash_Results"], sc_rows, "Super Cash")

    # BADGER5
    b5_ws = get_or_create_worksheet(ss, "Badger5_Results", rows=2000, cols=len(TABS_SCHEMA["Badger5_Results"]))
    b5_game = find_game("badger 5")
    b5_rows: List[List[Any]] = []
    if b5_game:
        for d in collect_draws_from_game(b5_game):
            date = normalize_date(d.get("date"))
            mains, _, _ = parse_numbers_objects(d.get("numbers", []))
            n1,n2,n3,n4,n5 = (mains + [None]*5)[:5]
            b5_rows.append([date, n1,n2,n3,n4,n5])
        b5_rows.sort(key=lambda r: (r[0] or ""), reverse=True)
    write_rows(b5_ws, TABS_SCHEMA["Badger5_Results"], b5_rows, "Badger 5")

    print("Success! Results loaded using v3.2 schema handler.")
    if sa_email:
        print(f"(If cells still won't update, allow editor access for {sa_email} on protected ranges.)")

if __name__ == "__main__":
    main()
