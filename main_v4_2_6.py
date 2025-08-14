#!/usr/bin/env python3
'''
main.py — v4.2.6
- Fix: corrected quoting that could break around `hist_text = "\n".join(...)` in some copy/paste cases.
- Guarantee totals with final filler; clearer debug logs; adaptive weighting; optional LLM; Run_Log gating.
'''
import os, json, datetime as dt, re, random
from typing import Any, Dict, List, Optional, Tuple

import requests
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError

# ---- Config ----
DEFAULT_SHEET_NAME = "Lottery Predictor New August 25"
API_URL_WI = "https://lottery-results.p.rapidapi.com/games-by-state/us/wi"
API_HOST = "lottery-results.p.rapidapi.com"

RESULT_SCHEMAS = {
    "Powerball_Results": ["Date","N1","N2","N3","N4","N5","PB"],
    "Megabucks_Results": ["Date","N1","N2","N3","N4","N5","N6"],
    "SuperCash_Results": ["Date","N1","N2","N3","N4","N5","N6","Doubler"],
    "Badger5_Results":   ["Date","N1","N2","N3","N4","N5"],
}
TRACKER_COLS = ["Timestamp","Game","Prediction","Method","Win Count","Matches","Match Count"]

# ---- Utilities ----
def fail(msg:str): raise RuntimeError(msg)

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
    max_rows = 5000
    ws_raw.update(values=[[ln] for ln in pretty[:max_rows]], range_name="A2")

# ---- Date helpers ----
def normalize_date(s: Any) -> str:
    s = "" if s is None else str(s).strip()
    if not s: return ""
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm,dd,yy = m.groups()
        try: return f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}"
        except: return s
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m: return s[:10]
    return s

def date_key(s: Any) -> Tuple[int,int,int,int]:
    ss = normalize_date(s)
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", ss)
    if m:
        y,mn,d = m.groups()
        return (int(y), int(mn), int(d), -10)
    m2 = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", str(s).strip())
    if m2:
        mm,dd,y = m2.groups()
        return (int(y), int(mm), int(dd), -9)
    return (0,0,0,-100)

def to_int(v)->Optional[int]:
    try:
        sv = str(v).strip()
        if sv == "": return None
        return int(sv)
    except: 
        return None

def parse_numbers_objects(nums: List[Dict[str,Any]])->Tuple[List[int], Optional[int], Optional[str]]:
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
            iv = to_int(val)
            if iv is not None: mains.append(iv)
    return mains, special, doubler

def top_level_games(data: Dict[str,Any]) -> List[Dict[str,Any]]:
    if not isinstance(data, dict): return []
    if isinstance(data.get("data"), list): return [g for g in data["data"] if isinstance(g, dict)]
    if isinstance(data.get("games"), list): return [g for g in data["games"] if isinstance(g, dict)]
    games = []
    for k,v in data.items():
        if str(k).isdigit() and isinstance(v, dict):
            games.append(v)
    return games

def collect_draws_from_game(game: Dict[str,Any]) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    plays = game.get("plays") if isinstance(game, dict) else None
    if not isinstance(plays, list): return out
    for p in plays:
        if not isinstance(p, dict): continue
        draws = p.get("draws")
        if not isinstance(draws, list): continue
        out.extend([d for d in draws if isinstance(d, dict)])
    return out

def col_letter(n:int)->str:
    s=""
    while n>0:
        n, r = divmod(n-1,26)
        s = chr(65+r)+s
    return s or "A"

# ---- Read/Write tables with normalization ----
def coerce_row_types(ws_title:str, headers: List[str], row: List[Any]) -> List[Any]:
    out = list(row) + [""]*(len(headers)-len(row))
    out[0] = normalize_date(out[0])
    if ws_title=="Powerball_Results":
        idxs = [1,2,3,4,5,6]
    elif ws_title=="Megabucks_Results":
        idxs = [1,2,3,4,5,6]
    elif ws_title=="SuperCash_Results":
        idxs = [1,2,3,4,5,6]
    elif ws_title=="Badger5_Results":
        idxs = [1,2,3,4,5]
    else:
        idxs = []
    for i in idxs:
        out[i] = to_int(out[i]) if str(out[i]).strip()!="" else None
    return out[:len(headers)]

def read_existing(ws, ws_title:str, expected_cols:int) -> List[List[Any]]:
    vals = ws.get_all_values()
    header = vals[0][:expected_cols] if vals else [""]*expected_cols
    body = vals[1:] if len(vals)>=2 else []
    fixed = [coerce_row_types(ws_title, header, r) for r in body]
    return fixed

def write_table(ws, headers: List[str], rows: List[List[Any]]):
    try: ws.update(values=[headers], range_name="A1")
    except APIError: pass
    ncols = len(headers); end_col = col_letter(ncols)
    if rows:
        ws.update(values=rows, range_name=f"A2:{end_col}{len(rows)+1}")
    else:
        ws.update(values=[["" for _ in range(ncols)]], range_name="A2")

def merge_by_date(existing: List[List[Any]], new_rows: List[List[Any]], date_idx:int=0) -> List[List[Any]]:
    m: Dict[str,List[Any]] = {}
    for r in existing:
        key = normalize_date(r[date_idx])
        if key: m[key] = r
    for r in new_rows:
        key = normalize_date(r[date_idx])
        if key and key not in m:
            m[key] = r
    rows = list(m.values())
    rows.sort(key=lambda r: date_key(r[date_idx]), reverse=True)
    return rows

# ---- Builders per game ----
def pb_rows_from_draws(draws: List[Dict[str,Any]]) -> List[List[Any]]:
    out=[]
    for d in draws:
        date = normalize_date(d.get("date"))
        mains, special, _ = parse_numbers_objects(d.get("numbers", []))
        n1,n2,n3,n4,n5 = (mains + [None]*5)[:5]
        out.append([date,n1,n2,n3,n4,n5,special])
    out.sort(key=lambda r: date_key(r[0]), reverse=True)
    return out

def mg_rows_from_draws(draws: List[Dict[str,Any]]) -> List[List[Any]]:
    out=[]
    for d in draws:
        date = normalize_date(d.get("date"))
        mains, _, _ = parse_numbers_objects(d.get("numbers", []))
        n1,n2,n3,n4,n5,n6 = (mains + [None]*6)[:6]
        out.append([date,n1,n2,n3,n4,n5,n6])
    out.sort(key=lambda r: date_key(r[0]), reverse=True)
    return out

def sc_rows_from_draws(draws: List[Dict[str,Any]]) -> List[List[Any]]:
    out=[]
    for d in draws:
        date = normalize_date(d.get("date"))
        mains, _, doubler = parse_numbers_objects(d.get("numbers", []))
        n1,n2,n3,n4,n5,n6 = (mains + [None]*6)[:6]
        out.append([date,n1,n2,n3,n4,n5,n6, (doubler or "")])
    out.sort(key=lambda r: date_key(r[0]), reverse=True)
    return out

def b5_rows_from_draws(draws: List[Dict[str,Any]]) -> List[List[Any]]:
    out=[]
    for d in draws:
        date = normalize_date(d.get("date"))
        mains, _, _ = parse_numbers_objects(d.get("numbers", []))
        n1,n2,n3,n4,n5 = (mains + [None]*5)[:5]
        out.append([date,n1,n2,n3,n4,n5])
    out.sort(key=lambda r: date_key(r[0]), reverse=True)
    return out

# ---- Methods (non‑LLM) ----
def unique_combo(nums: List[int], need:int, lo:int, hi:int) -> List[int]:
    s = sorted(set(n for n in nums if isinstance(n,int) and lo<=n<=hi))
    while len(s) < need:
        x = random.randint(lo, hi)
        if x not in s: s.append(x)
    return sorted(s)[:need]

def mutate(nums: List[int], lo:int, hi:int) -> List[int]:
    s = list(sorted(set(int(x) for x in nums if str(x).isdigit())))
    if not s: return s
    for _ in range(50):
        idx = random.randrange(0, len(s))
        cand = random.randint(lo, hi)
        if cand not in s:
            s[idx] = cand
            break
    return sorted(s)

def freq_method(rows: List[List[Any]], need:int, lo:int, hi:int, window:int=50) -> List[int]:
    hist = {}
    for r in rows[:window]:
        for v in r[1:1+need]:
            try:
                v = int(v); 
                if lo<=v<=hi:
                    hist[v] = hist.get(v, 0)+1
            except: pass
    ranked = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [k for k,_ in ranked][:need]
    return unique_combo(base, need, lo, hi)

def recency_method(rows: List[List[Any]], need:int, lo:int, hi:int, decay:float=0.9) -> List[int]:
    hist = {}
    w = 1.0
    for r in rows:
        for v in r[1:1+need]:
            try:
                v=int(v)
                if lo<=v<=hi: hist[v]=hist.get(v,0)+w
            except: pass
        w *= decay
    ranked = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [k for k,_ in ranked][:need]
    return unique_combo(base, need, lo, hi)

def markov1_method(rows: List[List[Any]], need:int, lo:int, hi:int) -> List[int]:
    seq = []
    for r in rows:
        for v in r[1:1+need]:
            try:
                v=int(v)
                if lo<=v<=hi: seq.append(v)
            except: pass
    trans = {i:{} for i in range(lo,hi+1)}
    for a,b in zip(seq, seq[1:]):
        trans[a][b] = trans[a].get(b,0)+1
    seed = []
    for r in rows[:3]:
        for v in r[1:1+need]:
            try:
                v=int(v)
                if lo<=v<=hi and v not in seed:
                    seed.append(v)
            except: pass
    cand = []
    for s in seed:
        nxt = sorted(trans.get(s, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        cand.extend([k for k,_ in nxt[:2]])
    return unique_combo(cand, need, lo, hi)

def pick_powerball(rows: List[List[Any]]) -> int:
    hist = {}
    w = 1.0
    for r in rows:
        try:
            pb = int(r[6]); hist[pb] = hist.get(pb,0)+w
        except: pass
        w *= 0.95
    if not hist: return random.randint(1,26)
    return sorted(hist.items(), key=lambda kv:(-kv[1], kv[0]))[0][0]

def fmt_prediction_str(game:str, nums: List[int], special: Optional[int]=None) -> str:
    core = "-".join(str(n) for n in nums)
    if game=="Powerball" and special is not None:
        return f"{core} (+PB {special})"
    return core

def intersect_count(a: List[int], b: List[int]) -> Tuple[List[int], int]:
    aset = set(a); bset = set(b)
    inter = sorted(list(aset & bset))
    return inter, len(inter)

def parse_prediction_to_nums(game:str, pred_str:str) -> Tuple[List[int], Optional[int]]:
    main_part = pred_str
    special = None
    if game=="Powerball" and "(+PB" in pred_str:
        main_part = pred_str.split("(+PB")[0].strip()
        try:
            special = int(re.findall(r"\(\+PB\s+(\d+)\)", pred_str)[0])
        except:
            special = None
    mains = []
    for tok in re.split(r"[-\s,]+", main_part):
        tok = tok.strip()
        if tok.isdigit(): mains.append(int(tok))
    return mains, special

# ---- Run_Log ----
def read_runlog(ss):
    ws = get_or_create_worksheet(ss, "Run_Log", rows=100, cols=4)
    vals = ws.get_all_values()
    if not vals:
        try: ws.update(values=[["Game","LastResultDate","LastPredictedNextDraw"]], range_name="A1")
        except APIError: pass
        return {}
    header = vals[0]
    rows = vals[1:]
    idx = {name:i for i,name in enumerate(header)}
    db = {}
    for r in rows:
        if not r or len(r)<1: continue
        gm = r[0].strip()
        last_res = r[idx.get("LastResultDate",1)] if len(r)>1 else ""
        last_pred = r[idx.get("LastPredictedNextDraw",2)] if len(r)>2 else ""
        db[gm] = {"LastResultDate": normalize_date(last_res), "LastPredictedNextDraw": normalize_date(last_pred)}
    return db

def write_runlog(ss, db):
    ws = get_or_create_worksheet(ss, "Run_Log", rows=100, cols=4)
    rows = [["Game","LastResultDate","LastPredictedNextDraw"]]
    for gm,info in db.items():
        rows.append([gm, normalize_date(info.get("LastResultDate","")), normalize_date(info.get("LastPredictedNextDraw",""))])
    ws.update(values=rows, range_name="A1")

# ---- Evaluate pending predictions ----
def evaluate_pending_predictions(ss, game:str, latest_row: List[Any]):
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    try: ws.update(values=[TRACKER_COLS], range_name="A1")
    except APIError: pass

    all_vals = ws.get_all_values()
    if not all_vals: return
    header = all_vals[0]; rows = all_vals[1:]
    col_idx = {name:i for i,name in enumerate(header)}

    if game=="Powerball":
        latest_mains = [int(x) for x in latest_row[1:6] if str(x).isdigit()]
    elif game=="Megabucks":
        latest_mains = [int(x) for x in latest_row[1:7] if str(x).isdigit()]
    elif game=="Super Cash":
        latest_mains = [int(x) for x in latest_row[1:7] if str(x).isdigit()]
    elif game=="Badger 5":
        latest_mains = [int(x) for x in latest_row[1:6] if str(x).isdigit()]
    else:
        latest_mains = []

    updates = []
    for i, r in enumerate(rows, start=2):
        try:
            gm = r[col_idx["Game"]].strip()
        except: 
            continue
        if gm != game: 
            continue
        matches = r[col_idx["Matches"]].strip() if len(r)>col_idx["Matches"] else ""
        if matches != "":
            continue
        pred_str = r[col_idx["Prediction"]]
        pred_nums, _ = parse_prediction_to_nums(game, pred_str)
        inter, cnt = intersect_count(pred_nums, latest_mains)
        win = 1 if (game=="Powerball" and cnt==5) or \
                  (game=="Megabucks" and cnt==6) or \
                  (game=="Super Cash" and cnt==6) or \
                  (game=="Badger 5" and cnt==5) else 0
        updates.append((i, {"Win Count": str(win), "Matches": ",".join(str(x) for x in inter), "Match Count": str(cnt)}))
    if updates:
        mod_rows = [list(r) + [""]*(len(TRACKER_COLS)-len(r)) for r in rows]
        for i, changes in updates:
            ridx = i-2
            for k,v in changes.items():
                c = col_idx[k]
                mod_rows[ridx][c] = v
        end_col = col_letter(len(TRACKER_COLS))
        ws.update(values=mod_rows, range_name=f"A2:{end_col}{len(mod_rows)+1}")

# ---- Adaptive weights ----
def adaptive_weights(ss, game:str) -> Dict[str,float]:
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    recs = ws.get_all_records()
    stats = {}
    for r in recs:
        if r.get("Game","").strip()!=game: continue
        md = r.get("Method","").strip()
        try: mc = int(str(r.get("Match Count","0")).strip() or "0")
        except: mc = 0
        try: wc = int(str(r.get("Win Count","0")).strip() or "0")
        except: wc = 0
        if md not in stats: stats[md] = {"cnt":0,"match_sum":0,"wins":0}
        stats[md]["cnt"] += 1
        stats[md]["match_sum"] += mc
        stats[md]["wins"] += wc
    weights = {}
    for md, s in stats.items():
        if s["cnt"]==0: continue
        score = (s["match_sum"]/s["cnt"]) + 5.0*s["wins"]
        weights[md] = score
    if not weights: return {}
    total = sum(v for v in weights.values() if v>0)
    if total<=0:
        return {k:1.0/len(weights) for k in weights}
    return {k:v/total for k,v in weights.items()}

# ---- LLM method (optional) ----
def llm_pick_numbers(game:str, rows_hist: List[List[Any]]) -> Tuple[List[int], Optional[int]]:
    if os.getenv("ENABLE_LLM_METHOD","0") not in ("1","true","True","YES","yes"):
        return [], None
    api_key = os.getenv("OPENAI_API_KEY","").strip()
    if not api_key:
        return [], None

    model = os.getenv("OPENAI_MODEL","gpt-4o-mini")
    hist_lines = []
    for r in rows_hist[:100]:
        date = str(r[0])
        if game=="Powerball":
            nums = [r[1],r[2],r[3],r[4],r[5]]; pb = r[6]
            hist_lines.append(f"{date}: {nums} PB:{pb}")
        elif game=="Megabucks":
            nums = [r[1],r[2],r[3],r[4],r[5],r[6]]
            hist_lines.append(f"{date}: {nums}")
        elif game=="Super Cash":
            nums = [r[1],r[2],r[3],r[4],r[5],r[6]]; d = r[7]
            hist_lines.append(f"{date}: {nums} D:{d}")
        elif game=="Badger 5":
            nums = [r[1],r[2],r[3],r[4],r[5]]
            hist_lines.append(f"{date}: {nums}")
    hist_text = "\\n".join(hist_lines)

    system = (
        "You generate lottery number-set predictions strictly in the required ranges. "
        "Return only JSON like {\"mains\":[...],\"special\":<int or null>} with no extra text."
    )

    if game=="Powerball":
        rule = "Pick 5 distinct mains between 1 and 69, and 1 Powerball between 1 and 26."
    elif game=="Megabucks":
        rule = "Pick 6 distinct mains between 1 and 49. No special ball."
    elif game=="Super Cash":
        rule = "Pick 6 distinct mains between 1 and 39. No special ball."
    elif game=="Badger 5":
        rule = "Pick 5 distinct mains between 1 and 31. No special ball."
    else:
        return [], None

    user = (
        f"Game: {game}\\n"
        f"Rules: {rule}\\n"
        f"Recent results (newest first):\\n{hist_text}\\n"
        "Respond ONLY with JSON: {\"mains\":[int,...],\"special\":null or int}"
    )

    mains, special = [], None
    try:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":user}
                ],
                temperature=0.7,
            )
            content = resp.choices[0].message.content.strip()
        except Exception:
            import openai
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":user}
                ],
                temperature=0.7,
            )
            content = resp["choices"][0]["message"]["content"].strip()

        js = json.loads(content)
        mains = [int(x) for x in js.get("mains",[]) if isinstance(x,(int,str)) and str(x).isdigit()]
        sp = js.get("special", None)
        special = int(sp) if (sp is not None and str(sp).isdigit()) else None
    except Exception:
        return [], None

    def clip_and_unique(nums, need, lo, hi):
        s = []
        for x in nums:
            try:
                xi = int(x)
                if lo<=xi<=hi and xi not in s:
                    s.append(xi)
            except: pass
        while len(s) < need:
            xi = random.randint(lo, hi)
            if xi not in s:
                s.append(xi)
        return sorted(s)[:need]

    if game=="Powerball":
        mains = clip_and_unique(mains, 5, 1, 69)
        if special is None or not (1<=special<=26):
            special = random.randint(1,26)
    elif game=="Megabucks":
        mains = clip_and_unique(mains, 6, 1, 49); special = None
    elif game=="Super Cash":
        mains = clip_and_unique(mains, 6, 1, 39); special = None
    elif game=="Badger 5":
        mains = clip_and_unique(mains, 5, 1, 31); special = None

    return mains, special

# ---- Per‑game totals ----
def safe_int(s: Optional[str]) -> Optional[int]:
    if s is None: return None
    ss = str(s).strip()
    if ss.isdigit(): return int(ss)
    m = re.match(r"^\s*(\d+)", ss)
    if m: return int(m.group(1))
    return None

def target_total_for_game(game:str, enabled_method_count:int) -> int:
    mapping = {
        "Powerball":  os.getenv("PREDICTIONS_POWERBALL"),
        "Megabucks":  os.getenv("PREDICTIONS_MEGABUCKS"),
        "Super Cash": os.getenv("PREDICTIONS_SUPERCASH"),
        "Badger 5":   os.getenv("PREDICTIONS_BADGER5"),
    }
    specific_raw = mapping.get(game)
    specific = safe_int(specific_raw)
    global_default = safe_int(os.getenv("PREDICTIONS_PER_GAME"))
    if specific is not None:
        return max(specific, enabled_method_count)
    if global_default is not None:
        return max(global_default, enabled_method_count)
    return enabled_method_count

# ---- Write predictions ----
def write_predictions_for_game(ss, game:str, rows_hist: List[List[Any]], next_draw_date:str):
    if os.getenv("ENABLE_PREDICTIONS","1") not in ("1","true","True","YES","yes"):
        return
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    try: ws.update(values=[TRACKER_COLS], range_name="A1")
    except APIError: pass

    existing = ws.get_all_records()
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    existing_keys = {(str(r.get("Timestamp",""))[:10], r.get("Game","").strip(), r.get("Method","").strip(), r.get("Prediction","").strip()) for r in existing}

    def append_rows(new_rows: List[List[str]]):
        if not new_rows: return
        start_row = len(existing) + 2
        end_row = start_row + len(new_rows) - 1
        end_col = col_letter(len(TRACKER_COLS))
        ws.update(values=new_rows, range_name=f"A{start_row}:{end_col}{end_row}")

    # Game ranges
    if game=="Powerball":
        need, lo, hi = 5, 1, 69
    elif game=="Megabucks":
        need, lo, hi = 6, 1, 49
    elif game=="Super Cash":
        need, lo, hi = 6, 1, 39
    elif game=="Badger 5":
        need, lo, hi = 5, 1, 31
    else:
        return

    # Base picks (one per enabled method)
    methods: Dict[str, Tuple[List[int], Optional[int]]] = {}
    if os.getenv("ENABLE_BASELINE","1") in ("1","true","True","YES","yes"):
        methods["last_draw_baseline"] = (rows_hist[0][1:1+need] if rows_hist else [], None)
    methods["freq50"] = (freq_method(rows_hist, need, lo, hi, window=50), None)
    methods["recency"] = (recency_method(rows_hist, need, lo, hi, decay=0.92), None)
    methods["markov1"] = (markov1_method(rows_hist, need, lo, hi), None)

    special_pb = pick_powerball(rows_hist) if game=="Powerball" else None

    mains_llm, special_llm = llm_pick_numbers(game, rows_hist)
    llm_ok = bool(mains_llm)
    if llm_ok:
        methods["llm_gpt"] = (mains_llm, special_llm if game=="Powerball" else None)

    enabled_methods = list(methods.keys())
    total_target = target_total_for_game(game, len(enabled_methods))

    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    written_preds = set()
    to_write = []

    def emit(md:str, nums:List[int], sp:Optional[int]):
        special = sp if (md=="llm_gpt" and sp is not None) else (special_pb if game=="Powerball" else None)
        pred_str = fmt_prediction_str(game, list(map(int, nums)) if nums else [], special)
        key = (today, game, md, pred_str)
        if key in existing_keys or pred_str in written_preds:
            return False
        to_write.append([timestamp, game, pred_str, md, "0", "", ""])
        written_preds.add(pred_str)
        return True

    # Pass 1: one per method
    for md in enabled_methods:
        nums, sp = methods[md]
        emit(md, nums, sp)

    # Debug env & target
    env_name = {
        "Powerball": "PREDICTIONS_POWERBALL",
        "Megabucks": "PREDICTIONS_MEGABUCKS",
        "Super Cash":"PREDICTIONS_SUPERCASH",
        "Badger 5":  "PREDICTIONS_BADGER5",
    }[game]
    print(f"[{game}] env {env_name}={os.getenv(env_name)} global PREDICTIONS_PER_GAME={os.getenv('PREDICTIONS_PER_GAME')} ENABLE_BASELINE={os.getenv('ENABLE_BASELINE')} LLM_OK={llm_ok}")
    print(f"[{game}] enabled_methods={enabled_methods} target_total={total_target} emitted_base={len(to_write)}")

    # Pass 2: extra variants by weights (no cap per method)
    extra = max(0, total_target - len(to_write))
    if extra > 0:
        weights = adaptive_weights(ss, game)
        if not weights:
            weights = {m: 1.0/len(enabled_methods) for m in enabled_methods}
        else:
            total_w = sum(weights.get(m,0.0) for m in enabled_methods)
            if total_w <= 0:
                weights = {m: 1.0/len(enabled_methods) for m in enabled_methods}
            else:
                weights = {m: weights.get(m,0.0)/total_w for m in enabled_methods}

        def weighted_pick(w:Dict[str,float])->str:
            r = random.random(); cum=0.0
            for k,v in w.items():
                cum += v
                if r <= cum: return k
            return next(iter(w))

        base_cache = {md: list(methods[md][0]) for md in enabled_methods}
        base_sp    = {md: (methods[md][1] if md=="llm_gpt" else None) for md in enabled_methods}
        tries = 0
        max_tries = 600
        while extra > 0 and tries < max_tries:
            tries += 1
            md = weighted_pick(weights)
            base_nums = base_cache.get(md, [])
            if not base_nums:
                continue
            variant = mutate(base_nums, lo, hi)
            sp = base_sp.get(md, None)
            if emit(md, variant, sp):
                extra -= 1

        # Fallback: round-robin over any non-empty base until we reach target or hit a safety cap
        if extra > 0:
            non_empty = [m for m,n in base_cache.items() if n]
            rr = 0
            safety = 0
            while extra > 0 and non_empty and safety < 800:
                safety += 1
                md = non_empty[rr % len(non_empty)]
                rr += 1
                variant = mutate(base_cache[md], lo, hi)
                sp = base_sp.get(md, None)
                if emit(md, variant, sp):
                    extra -= 1

        # Final filler: purely random unique combos to guarantee hitting target
        if extra > 0:
            print(f"[{game}] activating final filler: remaining={extra}")
            def fresh_combo(need:int, lo:int, hi:int)->List[int]:
                s = set()
                while len(s) < need:
                    s.add(random.randint(lo, hi))
                return sorted(s)

            safety2 = 0
            while extra > 0 and safety2 < 2000:
                safety2 += 1
                nums = fresh_combo(need, lo, hi)
                if emit("filler_random", nums, None):
                    extra -= 1

    print(f"[{game}] wrote {len(to_write)} rows (target_total={total_target})")
    append_rows(to_write)

# ---- Main orchestrator ----
def main():
    sheet_name, rapid_key, client, sa_email = load_runtime_config()
    ss = open_sheet(client, sheet_name)

    ws_health = get_or_create_worksheet(ss, "Health_Check", rows=100, cols=3)
    ws_raw    = get_or_create_worksheet(ss, "Raw_API_WI", rows=6000, cols=1)
    ws_index  = get_or_create_worksheet(ss, "Games_Index", rows=100, cols=3)
    append_health_check(ws_health)

    data = fetch_wi_results(rapid_key)
    write_raw(ws_raw, data)
    games = top_level_games(data)

    def find_game(needle:str)->Optional[Dict[str,Any]]:
        n = needle.lower()
        for g in games:
            name = (str(g.get("name","")) + " " + str(g.get("code",""))).lower()
            if n in name: return g
        return None

    def merge_and_write(game_label: str, ws_title: str, need: int, row_builder):
        ws = get_or_create_worksheet(ss, ws_title, rows=6000, cols=len(RESULT_SCHEMAS[ws_title]))
        existing = read_existing(ws, ws_title, expected_cols=len(RESULT_SCHEMAS[ws_title]))
        g = find_game(game_label.lower())
        draws = collect_draws_from_game(g) if g else []
        rows = row_builder(draws)
        merged = merge_by_date(existing, rows, date_idx=0)
        write_table(ws, RESULT_SCHEMAS[ws_title], merged)
        next_draw = normalize_date(draws[0].get("nextDrawDate")) if draws else ""
        return merged, next_draw

    merged_pb, next_pb = merge_and_write("Powerball",  "Powerball_Results", 5, pb_rows_from_draws)
    merged_mg, next_mg = merge_and_write("Megabucks",  "Megabucks_Results", 6, mg_rows_from_draws)
    merged_sc, next_sc = merge_and_write("Super Cash", "SuperCash_Results", 6, sc_rows_from_draws)
    merged_b5, next_b5 = merge_and_write("Badger 5",   "Badger5_Results",   5, b5_rows_from_draws)

    idx_rows = [[i+1, str(g.get("code","")), str(g.get("name",""))] for i,g in enumerate(games)]
    try: ws_index.update(values=[["#", "game_code", "game_name"]], range_name="A1")
    except APIError: pass
    ws_index.update(values=(idx_rows or [["", "", "No games"]]), range_name="A2")

    runlog = read_runlog(ss)

    results = [
        ("Powerball", merged_pb, next_pb),
        ("Megabucks", merged_mg, next_mg),
        ("Super Cash", merged_sc, next_sc),
        ("Badger 5", merged_b5, next_b5),
    ]

    for game_label, merged_rows, next_draw in results:
        if not merged_rows: 
            continue
        latest_date = normalize_date(merged_rows[0][0])
        rl = runlog.get(game_label, {"LastResultDate":"", "LastPredictedNextDraw":""})
        prev = normalize_date(rl.get("LastResultDate",""))
        is_new = (latest_date != prev)
        print(f"[{game_label}] latest_date={latest_date} prev={prev} is_new={is_new}")
        if is_new:
            evaluate_pending_predictions(ss, game_label, merged_rows[0])
            write_predictions_for_game(ss, game_label, merged_rows, next_draw)
            runlog[game_label] = {"LastResultDate": latest_date, "LastPredictedNextDraw": normalize_date(next_draw or "")}

    write_runlog(ss, runlog)
    print("Success! v4.2.6 finished.")
    
if __name__ == "__main__":
    main()
