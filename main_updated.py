#!/usr/bin/env python3
"""
main.py — v4.5.2 (per-next-draw budgeting + Chicago-day timestamps + tracker normalization)

Key changes in 4.5.2:
- Add optional backtest runner (RUN_BACKTEST) with safe defaults (no-write unless WRITE_BACKTEST=1).
- Fix crash in adaptive_weights when legacy rows had numbers in the "Method" column
  (cast to str; skip unknown methods).
- Normalize Prediction_Tracker to exactly 8 columns (A..H) BEFORE computing weights
  and BEFORE appending new rows, so header/width drift can't break reads.
- Keeps: per-draw budgeting, Chicago-time timestamps, SHEET_ID targeting,
  diagnostics, merge & dedupe, adaptive weighting, optional LLM method, etc.
"""
from predictor_core import markov_mc_method as seeded_markov_mc_method
import os, json, datetime as dt, re, random, math
from typing import Any, Dict, List, Optional, Tuple

import requests
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError
from zoneinfo import ZoneInfo  # Python 3.10+

# ---- Local timezone helpers (defaults to America/Chicago for display/heartbeats) ----
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/Chicago")

def env_true(name: str, default: bool = False) -> bool:
    """Parse common truthy/falsey env var values safely.

    Treats '0'/'false'/'no'/'off'/'': False and '1'/'true'/'yes'/'on': True.
    If unset or unrecognized, returns `default`.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in ("1", "true", "t", "yes", "y", "on"):
        return True
    if val in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return default

TZ = ZoneInfo(LOCAL_TZ)

# ---- Markov Monte Carlo tuning ----
# These environment variables allow you to tune the full Markov Monte Carlo
# method.  If unset, defaults mirror the original markov_predictor.py values.
try:
    MARKOV_TEMP = float(os.getenv("MARKOV_TEMP", "1.0"))
except Exception:
    MARKOV_TEMP = 1.0
try:
    MARKOV_SMOOTH = float(os.getenv("MARKOV_SMOOTH", "1.0"))
except Exception:
    MARKOV_SMOOTH = 1.0
MARKOV_RANDOM_SEED = os.getenv("MARKOV_RANDOM_SEED", "auto")

def now_local() -> dt.datetime:
    return dt.datetime.now(TZ)

def today_local_str() -> str:
    return now_local().strftime("%Y-%m-%d")

def timestamp_local_str() -> str:
    return now_local().strftime("%Y-%m-%d %H:%M:%S %Z")

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

# Tracker has 8 columns (A..H), with "Next Draw Date" at C
TRACKER_COLS = ["Timestamp","Game","Next Draw Date","Prediction","Method","Win Count","Matches","Match Count"]

# ---- Utilities ----
def fail(msg:str):
    raise RuntimeError(msg)

def load_runtime_config():
    sheet_name = os.getenv("SHEET_NAME", DEFAULT_SHEET_NAME)
    sheet_id   = os.getenv("SHEET_ID", "").strip()
    rapid_key  = os.getenv("RAPID_API_KEY")
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    miss = [k for k,v in {"RAPID_API_KEY":rapid_key, "GOOGLE_CREDS_JSON":creds_json}.items() if not v]
    if miss:
        fail("Missing required environment variables: " + ", ".join(miss))
    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError:
        fail("GOOGLE_CREDS_JSON is not valid JSON.")
    creds = Credentials.from_service_account_info(creds_dict, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    client = gspread.authorize(creds)
    return sheet_name, sheet_id, rapid_key, client, creds_dict.get("client_email")

def open_sheet(client, sheet_name, sheet_id):
    try:
        if sheet_id:
            print(f"[DIAG] Opening by SHEET_ID={sheet_id}")
            ss = client.open_by_key(sheet_id)
        else:
            print(f"[DIAG] Opening by SHEET NAME='{sheet_name}' (tip: set SHEET_ID for precision)")
            ss = client.open(sheet_name)
        return ss
    except gspread.SpreadsheetNotFound:
        if sheet_id:
            fail(f'Google Sheet ID "{sheet_id}" not found or not shared with service account.')
        else:
            fail(f'Google Sheet "{sheet_name}" not found or not shared with service account.')

def get_or_create_worksheet(ss, title, rows=1000, cols=26):
    try:
        return ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def append_health_check(ws_health):
    # Keep Health_Check in UTC as a neutral heartbeat
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    headers = ["Timestamp_UTC","Status"]
    try: ws_health.update(values=[headers], range_name="A1")
    except APIError: pass
    ws_health.append_row([now_utc,"OK"], value_input_option="RAW")

def fetch_wi_results(rapid_key:str)->Dict[str,Any]:
    h = {"x-rapidapi-key": rapid_key, "x-rapidapi-host": API_HOST}
    r = requests.get(API_URL_WI, headers=h, timeout=45)
    r.raise_for_status()
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

# ---- Read/Write normalization ----
def coerce_row_types(ws_title:str, headers: List[str], row: List[Any]) -> List[Any]:
    out = list(row) + [""]*(len(headers)-len(row))
    out[0] = normalize_date(out[0])  # "Date" for result tabs OR "Timestamp" for tracker—safe to leave
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

# ---- Methods (non-LLM) ----
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


def _recency_number_weights(rows: List[List[Any]], need:int, lo:int, hi:int, window:int=120, decay:float=0.92) -> Dict[int, float]:
    """Recency-weighted per-number weights from result history (newest-first rows)."""
    hist: Dict[int, float] = {n: 0.0 for n in range(lo, hi+1)}
    w = 1.0
    for r in rows[:window]:
        for v in r[1:1+need]:
            try:
                iv = int(v)
            except Exception:
                continue
            if lo <= iv <= hi:
                hist[iv] = hist.get(iv, 0.0) + w
        w *= decay
    # Add tiny uniform mass to avoid zeros
    eps = 1e-6
    for n in range(lo, hi+1):
        hist[n] = float(hist.get(n, 0.0)) + eps
    return hist

def _number_success_from_tracker_records(records: List[Dict[str, Any]], game:str, lo:int, hi:int, lookback:int=2000) -> Dict[int, float]:
    """Per-number empirical 'match' rate from tracker rows (Bayesian-smoothed).
    This lets us gently down-weight numbers that are predicted often but rarely show up in matches.
    """
    pred_ct: Dict[int, int] = {}
    hit_ct: Dict[int, int] = {}
    # Records are in sheet order (oldest->newest). Use tail(lookback) for speed.
    for r in (records[-lookback:] if len(records) > lookback else records):
        try:
            if str(r.get("Game","")).strip() != game:
                continue
            matches_raw = str(r.get("Matches","") if r.get("Matches","") is not None else "").strip()
            ms = matches_raw.upper()
            # Only learn from evaluated rows (ignore pending placeholders).
            if ms in ("", "-", "PENDING"):
                continue
            pred_str = str(r.get("Prediction","") if r.get("Prediction","") is not None else "").strip()
            if not pred_str:
                continue
            pred_nums, _ = parse_prediction_to_nums(game, pred_str)
            if not pred_nums:
                continue
            pred_nums = [n for n in pred_nums if lo <= n <= hi]
            for n in pred_nums:
                pred_ct[n] = pred_ct.get(n, 0) + 1
            match_nums = [int(x) for x in re.split(r"[,\s]+", matches_raw) if str(x).strip().isdigit()]
            match_nums = [n for n in match_nums if lo <= n <= hi]
            for n in match_nums:
                hit_ct[n] = hit_ct.get(n, 0) + 1
        except Exception:
            continue

    # Bayesian smoothing: (hits+1)/(pred+2) => safe for low counts
    out: Dict[int, float] = {}
    for n, p in pred_ct.items():
        h = hit_ct.get(n, 0)
        out[n] = (h + 1.0) / (p + 2.0)
    return out

def _combine_number_weights(recency_w: Dict[int,float], success_w: Dict[int,float], lo:int, hi:int, gamma:float=0.6) -> Dict[int,float]:
    """Combine history-recency weight with tracker-learned success weight (mild)."""
    w: Dict[int,float] = {}
    for n in range(lo, hi+1):
        base = float(recency_w.get(n, 1e-6))
        succ = float(success_w.get(n, 1.0))
        # gamma < 1 keeps this as a gentle adjustment, not a hard filter.
        w[n] = base * (succ ** gamma)
    # normalize not strictly necessary for relative sampling, but keep bounded
    s = sum(w.values()) or 1.0
    for n in w:
        w[n] /= s
    return w

def _weighted_choice_excluding(lo:int, hi:int, weights: Dict[int,float], exclude:set, rng: random.Random) -> int:
    cands = []
    ws = []
    for n in range(lo, hi+1):
        if n in exclude:
            continue
        cands.append(n)
        ws.append(float(weights.get(n, 0.0)))
    if not cands:
        # fallback
        return rng.randint(lo, hi)
    tot = sum(ws)
    if tot <= 0:
        return rng.choice(cands)
    return rng.choices(cands, weights=ws, k=1)[0]

def mutate_weighted(nums: List[int], lo:int, hi:int, weights: Dict[int,float], use_counts: Dict[int,int], rng: Optional[random.Random]=None, div_power: float=1.0) -> List[int]:
    """Mutate one element, choosing the replacement using weights and penalizing already-overused numbers."""
    rng = rng or random.Random()
    s = list(sorted(set(int(x) for x in (nums or []) if str(x).isdigit())))
    if not s:
        return s
    idx = rng.randrange(0, len(s))
    cur_set = set(s)
    # Penalize already-used numbers in this draw to diversify tickets.
    adj = dict(weights)
    for n, c in (use_counts or {}).items():
        if lo <= n <= hi and c > 0:
            adj[n] = adj.get(n, 0.0) / ((1.0 + c) ** div_power)
    cand = _weighted_choice_excluding(lo, hi, adj, exclude=cur_set, rng=rng)
    s[idx] = cand
    return sorted(s)

def fresh_combo_weighted(need:int, lo:int, hi:int, weights: Dict[int,float], use_counts: Dict[int,int], rng: Optional[random.Random]=None, div_power: float=1.0) -> List[int]:
    rng = rng or random.Random()
    chosen = []
    chosen_set = set()
    adj = dict(weights)
    for n, c in (use_counts or {}).items():
        if lo <= n <= hi and c > 0:
            adj[n] = adj.get(n, 0.0) / ((1.0 + c) ** div_power)
    while len(chosen) < need:
        n = _weighted_choice_excluding(lo, hi, adj, exclude=chosen_set, rng=rng)
        chosen.append(n)
        chosen_set.add(n)
    return sorted(chosen)

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

def markov1_method(rows: List[List[Any]], need: int, lo: int, hi: int) -> List[int]:
    """
    Build a simple first‑order Markov prediction based on historical draws.
    The old version collected candidates and then sorted them, which biased
    the result toward the smallest numbers.  Instead, we aggregate transition
    counts per candidate and choose the top 'need' numbers by frequency.
    """
    # Flatten main numbers into a sequence.
    seq = []
    for r in rows:
        for v in r[1:1+need]:
            try:
                iv = int(v)
            except Exception:
                continue
            if lo <= iv <= hi:
                seq.append(iv)

    # Build transition counts.
    trans = {i: {} for i in range(lo, hi+1)}
    for a, b in zip(seq, seq[1:]):
        trans[a][b] = trans[a].get(b, 0) + 1

    # Use recent draws as seeds.
    seeds = []
    for r in rows[:3]:
        for v in r[1:1+need]:
            try:
                iv = int(v)
            except Exception:
                continue
            if lo <= iv <= hi and iv not in seeds:
                seeds.append(iv)

    # Accumulate candidate counts from the seeds.
    candidate_counts = {}
    for s in seeds:
        for nxt, cnt in trans.get(s, {}).items():
            candidate_counts[nxt] = candidate_counts.get(nxt, 0) + cnt

    # Take the top 'need' numbers by count (ties broken by the number value).
    ranked = sorted(candidate_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [k for k, _ in ranked[:need]]

    # Fill with unique random values if we didn’t get enough candidates.
    base_set = set(base)
    while len(base_set) < need:
        x = random.randint(lo, hi)
        if x not in base_set:
            base_set.add(x)

    # Return the result sorted for consistency with other methods.
    return sorted(base_set)

# ---- Full Markov Monte Carlo method ----
def softmax(xs: List[float], temperature: float) -> List[float]:
    """Compute a softmax distribution over the list `xs` with the given temperature."""
    if temperature <= 0:
        temperature = 1e-6
    m = max(xs) if xs else 0.0
    exps = [math.exp((x - m) / temperature) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def build_transition_and_prior(history: List[List[int]], max_num: int, smooth: float) -> Tuple[List[List[float]], List[float]]:
    """Build a transition matrix T and prior distribution from historical draws."""
    T = [[0.0] * (max_num + 1) for _ in range(max_num + 1)]
    freq = [0.0] * (max_num + 1)
    for draw in history:
        for a in draw:
            freq[a] += 1.0
        for a in draw:
            for b in draw:
                if a != b:
                    T[a][b] += 1.0
    for i in range(1, max_num + 1):
        row_sum = sum(T[i])
        if row_sum > 0:
            T[i] = [x / row_sum for x in T[i]]
    prior = [0.0] + [freq[j] + smooth for j in range(1, max_num + 1)]
    return T, prior

def next_probs_from_last(last_set: List[int], T: List[List[float]], prior: List[float], temperature: float) -> List[float]:
    """Compute the next probability distribution given the last set."""
    max_num = len(prior) - 1
    scores = [0.0] * (max_num + 1)
    for j in range(1, max_num + 1):
        scores[j] = prior[j]
    for i in last_set:
        row = T[i]
        for j in range(1, max_num + 1):
            scores[j] += row[j]
    # Discard the zero index so indexing aligns with numbers 1..max_num
    return [0.0] + softmax(scores[1:], temperature)

def pick_weighted_without_replacement(cands: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    """Pick `k` elements from `cands` without replacement based on weights."""
    chosen: List[int] = []
    items = list(zip(cands, weights))
    total = sum(w for _, w in items) or 1.0
    items = [(c, (w / total if total > 0 else 1.0 / len(items))) for c, w in items]
    while len(chosen) < k and items:
        cs, ws = zip(*items)
        j = rng.choices(range(len(cs)), weights=ws, k=1)[0]
        chosen.append(cs[j])
        items.pop(j)
        total = sum(w for _, w in items) or 1.0
        if total > 0 and items:
            items = [(c, w / total) for c, w in items]
    return chosen

def markov_mc_method(rows: List[List[Any]], need: int, lo: int, hi: int, temp: float, smooth: float) -> List[int]:
    """Generate a prediction set using a full Markov Monte Carlo model.

    This uses the history of draws (`rows`) to build a transition matrix and prior
    distribution, then samples a set of `need` numbers from 1..`hi` weighted
    according to the Markov model.  The last set is taken from the most
    recent draw in `rows` if available.  If history is insufficient, it
    falls back to random unique numbers.
    """
    # Build history of draws as list of integer lists
    history: List[List[int]] = []
    for r in rows:
        draw: List[int] = []
        for v in r[1:1 + need]:
            try:
                iv = int(v)
                if lo <= iv <= hi:
                    draw.append(iv)
            except Exception:
                continue
        if len(draw) >= need:
            history.append(sorted(draw))
    if not history:
        # Fallback: random unique numbers
        return sorted(random.sample(range(lo, hi + 1), need))
    max_num = hi
    T, prior = build_transition_and_prior(history, max_num, smooth)
    # Use the most recent draw as the last set
    last_set = history[0]
    # Compute probabilities for next draw
    probs = next_probs_from_last(last_set, T, prior, temp)
    cands = list(range(1, max_num + 1))
    weights = probs[1:]
    rng = random.Random()
    # Seed RNG if user provided a seed
    try:
        if MARKOV_RANDOM_SEED not in ("auto", "", None):
            rng_seed = int(MARKOV_RANDOM_SEED)
            rng.seed(rng_seed)
    except Exception:
        pass
    pick = pick_weighted_without_replacement(cands, weights, need, rng)
    return sorted(pick)

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

# ---- Tracker normalization (ensure header & width A..H) ----
def normalize_tracker(ws):
    """Ensure Prediction_Tracker has our exact header and 8 columns for every row."""
    # Enforce header
    try: ws.update(values=[TRACKER_COLS], range_name="A1")
    except APIError: pass
    vals = ws.get_all_values()
    body = vals[1:] if len(vals) >= 2 else []
    if not body:
        return
    need_cols = len(TRACKER_COLS)
    changed = False
    norm_rows = []
    for r in body:
        if len(r) != need_cols:
            changed = True
        norm_rows.append((list(r) + [""]*(need_cols - len(r)))[:need_cols])
    if changed:
        end_col = col_letter(need_cols)
        ws.update(values=norm_rows, range_name=f"A2:{end_col}{len(norm_rows)+1}")

# ---- Evaluate pending predictions ----
def evaluate_pending_predictions(ss, game:str, latest_row: List[Any]):
    if predictions_dry_run():
        print(f"[{game}] PREDICTIONS_DRY_RUN=1 — skipping evaluate_pending_predictions.")
        return
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    # Normalize before reading/updating
    normalize_tracker(ws)

    all_vals = ws.get_all_values()
    if not all_vals:
        return

    rows = all_vals[1:]
    col_idx = {name:i for i,name in enumerate(TRACKER_COLS)}

    # Determine the latest winning mains for this game
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
    for i, r in enumerate(rows, start=2):  # sheet row numbers
        r = (list(r) + [""]*(len(TRACKER_COLS) - len(r)))[:len(TRACKER_COLS)]

        try:
            gm = str(r[col_idx["Game"]]).strip()
        except Exception:
            continue
        if gm != game:
            continue        # Skip if already evaluated.
        # NOTE: Older versions (and some external scorers) may mark pending rows as '-' or 'PENDING'.
        # Treat those as *not* evaluated so they get rescored when results arrive.
        matches_cell = r[col_idx["Matches"]]
        ms = str(matches_cell).strip().upper()
        if ms not in ("", "-", "PENDING"):
            continue

        pred_str = r[col_idx["Prediction"]]
        pred_nums, _ = parse_prediction_to_nums(game, pred_str)
        inter = sorted(set(pred_nums) & set(latest_mains))
        cnt = len(inter)

        win = 1 if (game=="Powerball" and cnt==5) or \
                  (game=="Megabucks" and cnt==6) or \
                  (game=="Super Cash" and cnt==6) or \
                  (game=="Badger 5" and cnt==5) else 0

        r[col_idx["Win Count"]]   = str(win)
        r[col_idx["Matches"]]     = ",".join(str(x) for x in inter)
        r[col_idx["Match Count"]] = str(cnt)

        updates.append((i, r))

    if updates:
        # Overwrite the range A2:H with normalized rows so the sheet width stays consistent
        mod_rows = []
        for r in rows:
            norm = (list(r) + [""]*(len(TRACKER_COLS)-len(r)))[:len(TRACKER_COLS)]
            mod_rows.append(norm)
        end_col = col_letter(len(TRACKER_COLS))  # 'H'
        ws.update(values=mod_rows, range_name=f"A2:{end_col}{len(mod_rows)+1}")

# ---- Adaptive weights ----
def adaptive_weights(ss, game:str) -> Dict[str,float]:
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    # Normalize before reading
    normalize_tracker(ws)
    recs = ws.get_all_records()
    stats = {}
    # When computing weights we treat `markov1` results as coming from
    # our new `markov_mc` method.  This preserves the historical
    # contribution of the old Markov implementation so the adaptive
    # weighting continues to favor Markov‑like predictions as before.
    # Only the supported methods are included here; `markov1` is
    # intentionally omitted so that its tallies roll into `markov_mc`.
    allowed_methods = {        "freq50",
        "recency",
        "markov_mc",
        "llm_gpt",
        "filler_random",
    }
    for r in recs:
        if str(r.get("Game","")).strip()!=game: 
            continue
        md = str(r.get("Method","") if r.get("Method","") is not None else "").strip()
        # Map legacy `markov1` method names to `markov_mc` so that
        # historical performance is credited to the new Monte Carlo version.
        if md == "markov1":
            md = "markov_mc"
        if md not in allowed_methods:
            # ignore legacy/misaligned rows
            continue
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
    if os.getenv("ENABLE_LLM_METHOD","0").lower() not in ("1","true","yes"):
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
    hist_text = "\n".join(hist_lines)

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
        f"Game: {game}\n"
        f"Rules: {rule}\n"
        f"Recent results (newest first):\n{hist_text}\n"
        'Respond ONLY with JSON: {"mains":[int,...],"special":null or int}'
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

# ---- Per-game totals ----
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

# ---- Write predictions (PER-DRAW budgeting) ----

# ---- Feature flags / helpers (non-breaking) ----

# Backtest flags (default OFF; safe/no-write by default)
RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0").lower() in ("1","true","yes")
WRITE_BACKTEST = os.getenv("WRITE_BACKTEST", "0").lower() in ("1","true","yes")

def disabled_methods_for_game(game:str) -> set:
    env_map = {
        "Powerball":  "DISABLE_METHODS_POWERBALL",
        "Megabucks":  "DISABLE_METHODS_MEGABUCKS",
        "Super Cash": "DISABLE_METHODS_SUPERCASH",
        "Badger 5":   "DISABLE_METHODS_BADGER5",
    }
    env = env_map.get(game, "")
    raw_global = os.getenv("DISABLE_METHODS", "") or ""
    raw_game = os.getenv(env, "") or ""
    raw = ",".join([raw_global, raw_game])
    return {m.strip() for m in raw.split(",") if m.strip()}

def predictions_dry_run() -> bool:
    return os.getenv("PREDICTIONS_DRY_RUN","0").lower() in ("1","true","yes")

def write_predictions_for_game(ss, game:str, rows_hist: List[List[Any]], next_draw_date:str):
    if os.getenv("ENABLE_PREDICTIONS","1").lower() not in ("1","true","yes"):
        return
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    # Normalize BEFORE reading/weighting/appending
    normalize_tracker(ws)

    existing = ws.get_all_records()

    # Use PER-DRAW key (fallback to today's local date if API omits)
    next_key = normalize_date(next_draw_date) or today_local_str()

    # Filter existing predictions for THIS (Game, Next Draw Date)
    existing_for_draw = [r for r in existing
                         if str(r.get("Game","")).strip()==game
                         and normalize_date(r.get("Next Draw Date",""))==next_key]
    existing_preds_for_draw = {str(r.get("Prediction","")).strip() for r in existing_for_draw}

    # Compute remaining to emit to hit target for *this draw*
    enable_baseline = str(os.getenv("ENABLE_BASELINE", "1")).strip().lower() in ("1","true","yes","y","on")
    enable_llm = (str(os.getenv("ENABLE_LLM_METHOD", "0")).strip().lower() in ("1","true","yes","y","on")
                  and bool((os.getenv("OPENAI_API_KEY","") or "").strip()))
    enabled_methods_count = (3 if enable_baseline else 0) + (1 if enable_llm else 0)
    total_target = target_total_for_game(game, enabled_methods_count)
    remaining = max(0, total_target - len(existing_for_draw))
    if remaining == 0:
        print(f"[{game}] Already have {len(existing_for_draw)} predictions for draw={next_key}; target={total_target}. Skipping new emissions.")
        return

    def append_rows(new_rows: List[List[str]]):
        if not new_rows: return
        # Re-read count in case someone edited between normalize and now
        existing2 = ws.get_all_values()
        start_row = (len(existing2)-1) + 2 if existing2 else 2  # rows after header
        end_row = start_row + len(new_rows) - 1
        end_col = col_letter(len(TRACKER_COLS))  # 'H'
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
    # --- Guard data: capture the latest official results to block exact repeats ---
    latest_mains = []
    latest_special = None
    if rows_hist:
        try:
            latest_row = rows_hist[0]
            latest_mains = [int(x) for x in latest_row[1:1+need] if str(x).strip().isdigit()]
            if game == "Powerball" and len(latest_row) > 6 and str(latest_row[6]).strip().isdigit():
                latest_special = int(latest_row[6])
        except Exception:
            latest_mains = []
            latest_special = None
    # -----------------------------------------------------------------------------

    # Determine a default special pick for Powerball based on history.  Earlier
    # versions of this script referenced a `special_pb` variable without
    # defining it, leading to a NameError.  Compute it once here so that
    # baseline and weighted variants have a consistent special ball value.  For
    # other games the special ball is always None.
    if game == "Powerball":
        try:
            special_pb: Optional[int] = pick_powerball(rows_hist)
        except Exception:
            special_pb = None
    else:
        special_pb = None


    # ---- Diversity / learning weights (optional, safe defaults) ----
    diversify_on = os.getenv("DIVERSIFY_PREDICTIONS", "1").lower() in ("1","true","yes")
    diversify_power = float(os.getenv("DIVERSIFY_POWER", "1.0") or "1.0")
    learn_on = os.getenv("LEARN_NUMBER_SUCCESS", "1").lower() in ("1","true","yes")
    # Initialize per-number usage counts from existing predictions for this draw
    use_counts: Dict[int,int] = {}
    for ps in existing_preds_for_draw:
        try:
            ns, _sp = parse_prediction_to_nums(game, ps)
            for n in ns:
                if lo <= n <= hi:
                    use_counts[n] = use_counts.get(n, 0) + 1
        except Exception:
            continue

    rec_w = _recency_number_weights(rows_hist, need, lo, hi, window=120, decay=0.92)
    succ_w = _number_success_from_tracker_records(existing, game, lo, hi, lookback=2000) if learn_on else {}
    num_w = _combine_number_weights(rec_w, succ_w, lo, hi, gamma=float(os.getenv("LEARN_GAMMA","0.6") or "0.6"))
    rng_local = random.Random()



    # Base picks (one per enabled method)
    # Base picks (one per enabled method)
    methods: Dict[str, Tuple[List[int], Optional[int]]] = {}
    # Baseline methods (freq/recency/markov) are optional as a group.
    if enable_baseline:
        methods["freq50"]  = (freq_method(rows_hist, need, lo, hi, window=50), None)
        methods["recency"] = (recency_method(rows_hist, need, lo, hi, decay=0.92), None)
        # Markov Monte Carlo method (seeded)
        methods["markov_mc"] = (seeded_markov_mc_method(rows_hist, need, lo, hi, MARKOV_TEMP, MARKOV_SMOOTH, seed_mode="auto"), None)

    # LLM method (optional)
    mains_llm, special_llm = llm_pick_numbers(game, rows_hist)
    llm_ok = bool(mains_llm)
    if llm_ok and enable_llm:
        methods["llm_gpt"] = (mains_llm, special_llm if game == "Powerball" else None)

    disabled = disabled_methods_for_game(game)

    # Apply per-game disables (if any)
    for _m in list(methods.keys()):
        if _m in disabled:
            methods.pop(_m, None)

    # Safety fallback: never leave a game with no methods
    if not methods:
        methods["markov_mc"] = (seeded_markov_mc_method(rows_hist, need, lo, hi, MARKOV_TEMP, MARKOV_SMOOTH, seed_mode="auto"), None)
    enabled_methods = list(methods.keys())
    if predictions_dry_run():
        print(f"[{game}] PREDICTIONS_DRY_RUN=1 — will not append predictions; exiting write_predictions_for_game.")
        return
    timestamp = timestamp_local_str()

    written_preds = set()
    to_write = []

    def emit(md:str, nums:List[int], sp:Optional[int]):
        special = sp if (md=="llm_gpt" and sp is not None) else (special_pb if game=="Powerball" else None)
        # Never emit baseline into projections (it is only a diagnostic reference)
        if md == "last_draw_baseline":
            return False
        # --- Block exact repeat of last official winning set ---
        try:
            nums_int = [int(n) for n in (nums or [])]
        except Exception:
            nums_int = nums or []
        if latest_mains:
            same_mains = sorted(nums_int) == sorted(latest_mains)
            if game == "Powerball":
                same_special = (special == latest_special) if latest_special is not None else True
                if same_mains and same_special:
                    return False
            else:
                if same_mains:
                    return False
        pred_str = fmt_prediction_str(game, list(map(int, nums)) if nums else [], special)
        if pred_str in existing_preds_for_draw or pred_str in written_preds:
            return False
        # Row: Timestamp | Game | Next Draw Date | Prediction | Method | Win Count | Matches | Match Count
        to_write.append([timestamp, game, next_key, pred_str, md, "0", "", ""])
        written_preds.add(pred_str)
        # Track number usage within this draw so later variants diversify
        try:
            for n in nums_int:
                if lo <= int(n) <= hi:
                    use_counts[int(n)] = use_counts.get(int(n), 0) + 1
        except Exception:
            pass
        return True

    # Pass 1: one per method
    for md in enabled_methods:
        if md.startswith("llm_"):
            continue
        if remaining <= 0: break
        nums, sp = methods[md]
        if emit(md, nums, sp):
            remaining -= 1

    env_name = {
        "Powerball": "PREDICTIONS_POWERBALL",
        "Megabucks": "PREDICTIONS_MEGABUCKS",
        "Super Cash":"PREDICTIONS_SUPERCASH",
        "Badger 5":  "PREDICTIONS_BADGER5",
    }[game]
    print(f"[{game}] env {env_name}={os.getenv(env_name)} global PREDICTIONS_PER_GAME={os.getenv('PREDICTIONS_PER_GAME')} ENABLE_BASELINE={os.getenv('ENABLE_BASELINE')} LLM_OK={llm_ok}")
    print(f"[{game}] existing_for_draw={len(existing_for_draw)} target_total={total_target} remaining_after_base={remaining} base_emitted={len(to_write)} draw={next_key}")

    # Pass 2: extra variants by weights (no cap per method)
    if remaining > 0:
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
        # Ensure the raw LLM pick is included at least once (LLM is skipped in Pass 1 by design)
        if remaining > 0 and "llm_gpt" in enabled_methods:
            _base_nums = base_cache.get("llm_gpt", [])
            _sp = base_sp.get("llm_gpt", None)
            if _base_nums and emit("llm_gpt", _base_nums, _sp):
                remaining -= 1

        tries = 0
        max_tries = 800
        while remaining > 0 and tries < max_tries:
            tries += 1
            md = weighted_pick(weights)
            base_nums = base_cache.get(md, [])
            if not base_nums:
                continue
            variant = mutate_weighted(base_nums, lo, hi, num_w, use_counts, rng=rng_local, div_power=diversify_power) if diversify_on else mutate(base_nums, lo, hi)
            sp = base_sp.get(md, None)
            if emit(md, variant, sp):
                remaining -= 1

        # Fallback: round-robin
        if remaining > 0:
            non_empty = [m for m,n in base_cache.items() if n]
            rr = 0
            safety = 0
            while remaining > 0 and non_empty and safety < 1000:
                safety += 1
                md = non_empty[rr % len(non_empty)]
                rr += 1
                variant = mutate_weighted(base_cache[md], lo, hi, num_w, use_counts, rng=rng_local, div_power=diversify_power) if diversify_on else mutate(base_cache[md], lo, hi)
                sp = base_sp.get(md, None)
                if emit(md, variant, sp):
                    remaining -= 1

        # Final filler: random unique combos
        if remaining > 0:
            print(f"[{game}] activating final filler: need {remaining} more (draw={next_key})")
            def fresh_combo(need:int, lo:int, hi:int)->List[int]:
                if diversify_on:
                    return fresh_combo_weighted(need, lo, hi, num_w, use_counts, rng=rng_local, div_power=diversify_power)
                s = set()
                while len(s) < need:
                    s.add(random.randint(lo, hi))
                return sorted(s)

            safety2 = 0
            while remaining > 0 and safety2 < 3000:
                safety2 += 1
                nums = fresh_combo(need, lo, hi)
                if emit("filler_random", nums, None):
                    remaining -= 1

    print(f"[{game}] appended {len(to_write)} rows this run for draw={next_key}; now should total {len(existing_for_draw)+len(to_write)} (target={total_target})")
    append_rows(to_write)

# ---- Main orchestrator ----
def main():
    sheet_name, sheet_id, rapid_key, client, sa_email = load_runtime_config()

    # Open spreadsheet + diagnostics
    ss = open_sheet(client, sheet_name, sheet_id)
    try:
        ss_url = ss.url
    except Exception:
        ss_url = f"https://docs.google.com/spreadsheets/d/{getattr(ss, 'id', '<id>')}/edit"
    print(f"[DIAG] Service account email: {sa_email}")
    print(f"[DIAG] Spreadsheet title: {ss.title} | id: {ss.id}")
    print(f"[DIAG] Spreadsheet URL: {ss_url}")
    print(f"[DIAG] LOCAL_TZ in effect: {LOCAL_TZ}")

    # Health & scratch
    ws_health = get_or_create_worksheet(ss, "Health_Check", rows=100, cols=3)
    ws_raw    = get_or_create_worksheet(ss, "Raw_API_WI", rows=6000, cols=1)
    ws_index  = get_or_create_worksheet(ss, "Games_Index", rows=100, cols=3)
    append_health_check(ws_health)

    # Touch log to prove writes every run (use local tz for visibility)
    try:
        ws_touch = get_or_create_worksheet(ss, "Debug_Touch", rows=200, cols=3)
        ws_touch.append_row([timestamp_local_str(), "ran", "v4.5.1"], value_input_option="RAW")
        print("[DIAG] Wrote a Debug_Touch row successfully.")
    except Exception as e:
        print(f"[DIAG] Failed to write Debug_Touch: {e}")

    # Fetch & parse
    data = fetch_wi_results(rapid_key)
    write_raw(ws_raw, data)
    games = top_level_games(data)

    def find_game(needle:str)->Optional[Dict[str,Any]]:
        n = needle.lower()
        for g in games:
            name = (str(g.get("name","")) + " " + str(g.get("code",""))).lower()
            if n in name: return g
        return None

    def collect_next_draw_date(draws: List[Dict[str,Any]]) -> str:
        if not draws: return ""
        try:
            nd = draws[0].get("nextDrawDate")
            return normalize_date(nd)
        except Exception:
            return ""

    def merge_and_write(game_label: str, ws_title: str, need: int, row_builder):
        ws = get_or_create_worksheet(ss, ws_title, rows=6000, cols=len(RESULT_SCHEMAS[ws_title]))
        existing = read_existing(ws, ws_title, expected_cols=len(RESULT_SCHEMAS[ws_title]))
        g = find_game(game_label.lower())
        draws = collect_draws_from_game(g) if g else []
        rows = row_builder(draws)

        api_latest = rows[0][0] if rows else ""
        print(f"[{game_label}] API latest date: {api_latest}  existing_rows={len(existing)} new_rows={len(rows)}")

        merged = merge_by_date(existing, rows, date_idx=0)
        write_table(ws, RESULT_SCHEMAS[ws_title], merged)

        min_date = merged[-1][0] if merged else ""
        max_date = merged[0][0] if merged else ""
        print(f"[{game_label}] merged_rows={len(merged)} date_range={min_date} .. {max_date}  -> sheet_tab='{ws_title}'")

        return merged, collect_next_draw_date(draws)

    merged_pb, next_pb = merge_and_write("Powerball",  "Powerball_Results", 5, pb_rows_from_draws)
    merged_mg, next_mg = merge_and_write("Megabucks",  "Megabucks_Results", 6, mg_rows_from_draws)
    merged_sc, next_sc = merge_and_write("Super Cash", "SuperCash_Results", 6, sc_rows_from_draws)
    merged_b5, next_b5 = merge_and_write("Badger 5",   "Badger5_Results",   5, b5_rows_from_draws)


    # Optional read-only backtest for Powerball: frequency+recency hit rate.
    if RUN_BACKTEST:
        try:
            print("[BACKTEST] Enabled (RUN_BACKTEST=1). Running Powerball frequency+recency backtest (read-only by default).")
            bt = backtest_powerball_frequency_recency(merged_pb, lookback=200)
            if WRITE_BACKTEST:
                write_backtest(ss, bt)
        except Exception as e:
            print(f"[BACKTEST] Failed safely (continuing normal run): {e}")

    # Write/update a small index sheet of game codes/names (helps debugging + sheet navigation).
    try:
        ws_index = ss.worksheet("Games_Index")
    except Exception:
        ws_index = ss.add_worksheet(title="Games_Index", rows=200, cols=10)
    ws_index.update(values=[["#", "code", "name"]], range_name="A1")
    idx_rows = [[i+1, str(g.get("code", "")), str(g.get("name", ""))] for i, g in enumerate(games)]
    ws_index.update(values=(idx_rows or [["", "", "No games"]]), range_name="A2")

    runlog = read_runlog(ss)

    results = [
        ("Powerball", merged_pb, next_pb),
        ("Megabucks", merged_mg, next_mg),
        ("Super Cash", merged_sc, next_sc),
        ("Badger 5", merged_b5, next_b5),
    ]

    force = os.getenv("FORCE_PREDICT_TODAY", "0").lower() in ("1","true","yes")

    # Emit new predictions ONLY when a new result arrived (or when forced),
    # and budget per (Game, NextDrawDate).
    for game_label, merged_rows, next_draw in results:
        if not merged_rows:
            continue
        latest_date = normalize_date(merged_rows[0][0])
        rl = runlog.get(game_label, {"LastResultDate":"", "LastPredictedNextDraw":""})
        prev = normalize_date(rl.get("LastResultDate",""))
        is_new = (latest_date != prev)
        print(f"[{game_label}] latest_date={latest_date} prev={prev} is_new={is_new} force={force} next_draw={normalize_date(next_draw) or today_local_str()}")
        if is_new or force:
            if is_new:
                evaluate_pending_predictions(ss, game_label, merged_rows[0])
                runlog[game_label] = {
                    "LastResultDate": latest_date,
                    "LastPredictedNextDraw": normalize_date(next_draw or "")
                }
            write_predictions_for_game(ss, game_label, merged_rows, next_draw)

    write_runlog(ss, runlog)
    print("Success! v4.5.1 finished. See Debug_Touch for this run’s timestamp.")

if __name__ == "__main__":
    main()
