import os, time, random, numpy as np

_seed_raw = os.environ.get("MARKOV_RANDOM_SEED","").strip()
if _seed_raw.lower() == "auto" or _seed_raw == "":
    seed = int(time.time()) ^ (os.getpid() << 16)
else:
    try:
        seed = int(_seed_raw)
    except ValueError:
        seed = 42
np.random.seed(seed); random.seed(seed)
print(f"[markov_mc] RNG seed={seed} (raw='{_seed_raw or 'auto'}')")

#!/usr/bin/env python3
import os, json, math, random
from collections import Counter, defaultdict
from typing import List, Tuple, Optional
from datetime import datetime
import pytz
import pandas as pd

import gspread
from google.oauth2.service_account import Credentials

# ---------- Env helpers ----------
def env_bool(name: str, default: bool=False) -> bool:
    v = os.environ.get(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name); 
    try:
        return int(str(v)) if v is not None and str(v).strip()!="" else default
    except: 
        return default

def env_float(name: str, default: float) -> float:
    v = os.environ.get(name); 
    try:
        return float(str(v)) if v is not None and str(v).strip()!="" else default
    except: 
        return default

# ---------- Math helpers ----------
def softmax(xs: List[float], temperature: float=1.0) -> List[float]:
    if temperature <= 0:
        m = max(xs); onehot = [0.0]*len(xs); onehot[xs.index(m)] = 1.0; return onehot
    scaled = [x/temperature for x in xs]
    m = max(scaled)
    exps = [math.exp(x-m) for x in scaled]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def pick_weighted_without_replacement(cands: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    chosen = []
    items = list(zip(cands, weights))
    # normalize once
    total = sum(w for _,w in items) or 1.0
    items = [(c, (w/total if total>0 else 1.0/len(items))) for c,w in items]
    while len(chosen) < k and items:
        cs, ws = zip(*items)
        j = rng.choices(range(len(cs)), weights=ws, k=1)[0]
        chosen.append(cs[j])
        items.pop(j)
        total = sum(w for _,w in items) or 1.0
        items = [(c, (w/total if total>0 else 1.0/len(items))) for c,w in items]
    return chosen

# ---------- Markov model ----------
def build_markov(history_sets: List[List[int]], max_num: int, smooth: float=1.0) -> Tuple[List[List[float]], List[float]]:
    T = [[0.0]*(max_num+1) for _ in range(max_num+1)]
    freq = [0.0]*(max_num+1)
    prev = None
    for s in history_sets:
        for n in s:
            freq[n] += 1.0
        if prev is not None:
            for i in prev:
                for j in s:
                    T[i][j] += 1.0
        prev = s
    # smoothing
    for i in range(1, max_num+1):
        for j in range(1, max_num+1):
            T[i][j] += smooth
    prior = [0.0] + [freq[j] + smooth for j in range(1, max_num+1)]
    return T, prior

def next_probs_from_last(last_set: List[int], T: List[List[float]], prior: List[float], temperature: float) -> List[float]:
    max_num = len(prior)-1
    scores = [0.0]*(max_num+1)
    # base prior
    for j in range(1, max_num+1):
        scores[j] = prior[j]
    # transitions influence
    for i in last_set:
        row = T[i]
        for j in range(1, max_num+1):
            scores[j] += row[j]
    return [0.0] + softmax(scores[1:], temperature)

# ---------- Sheets IO ----------
def open_sheet():
    sheet_name = os.environ["SHEET_NAME"]
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    creds_dict = json.loads(creds_json)
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    gc = gspread.authorize(creds)
    return gc.open(sheet_name)

def detect_results_ws(sh, game: str):
    candidates = [
        f"{game}_Results",
        f"{game} Results",
        game.replace(" ", "") + "_Results",
        game,
    ]
    for t in candidates:
        try:
            return sh.worksheet(t)
        except Exception:
            pass
    # heuristic search
    for ws in sh.worksheets():
        name = ws.title.lower()
        if game.split()[0].lower() in name and "result" in name:
            return ws
    raise RuntimeError(f"Could not find results tab for game '{game}'.")

def detect_numbers_frame(values):
    if not values: raise RuntimeError("Empty worksheet")
    header = [str(h).strip() for h in values[0]]
    df = pd.DataFrame(values[1:], columns=header)
    # pick date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    # detect numeric cols
    def looks_int(x):
        try:
            if x is None or str(x).strip()=="": return False
            int(float(str(x).strip()));
            return True
        except: return False
    number_cols = []
    for c in df.columns:
        if c == date_col: continue
        vals = [v for v in df[c].tolist() if v not in (None,""," ")]
        if not vals: continue
        ints = sum(1 for v in vals if looks_int(v))
        if len(vals)>0 and (ints/len(vals))>=0.9:
            number_cols.append(c)
    # clean
    def to_dt(x):
        try: return pd.to_datetime(x).date()
        except: return pd.NaT
    df = df[[date_col]+number_cols].dropna(how="all")
    df[date_col] = df[date_col].apply(to_dt)
    df = df.dropna(subset=[date_col])
    for c in number_cols:
        df[c] = df[c].apply(lambda x: int(float(x)) if str(x).strip() not in ("","nan","None") else None)
    df = df.dropna(subset=number_cols)
    df[number_cols] = df[number_cols].astype(int)
    df = df.sort_values(by=[date_col]).reset_index(drop=True)
    return df, date_col, number_cols

def ensure_predictions_ws(sh, game: str, n_numbers: int):
    title = f"{game}_Predictions"
    try:
        ws = sh.worksheet(title)
    except Exception:
        ws = sh.add_worksheet(title=title, rows=5000, cols=20)
        # add header
        header = ["Timestamp", "Method"] + [f"N{i+1}" for i in range(n_numbers)]
        ws.update("A1", [header])
    # ensure header exists
    vals = ws.get_all_values()
    if not vals:
        header = ["Timestamp", "Method"] + [f"N{i+1}" for i in range(n_numbers)]
        ws.update("A1", [header])
    return ws

# ---------- Core prediction ----------
def predict_for_game(sh, game: str, emit_count: int, temp: float, smooth: float, sims: int, rng: random.Random, method_name: str, unique_only: bool=True):
    # read results
    ws = detect_results_ws(sh, game)
    values = ws.get_all_values()
    df, date_col, num_cols = detect_numbers_frame(values)
    draws = [sorted([int(df.at[i, c]) for c in num_cols]) for i in range(len(df))]
    n_numbers = len(num_cols)
    max_num = max(df[c].max() for c in num_cols)
    last = draws[-1]
    # build model
    T, prior = build_markov(draws, max_num=max_num, smooth=smooth)
    # Monte Carlo bagging
    bag = Counter()
    cands_all = list(range(1, max_num+1))
    for _ in range(sims):
        probs = next_probs_from_last(last, T, prior, temp)
        weights = [probs[j] for j in cands_all]
        picks = pick_weighted_without_replacement(cands_all, weights, n_numbers, rng)
        bag[tuple(sorted(picks))] += 1
    # select top unique combos
    unique = []
    seen = set()
    for tpl, _freq in bag.most_common():
        if unique_only and tpl in seen: 
            continue
        seen.add(tpl)
        unique.append(list(tpl))
        if len(unique) >= emit_count: break
    # write
    pred_ws = ensure_predictions_ws(sh, game, n_numbers)
    tz = pytz.timezone(os.environ.get("LOCAL_TZ", "America/Chicago"))
    ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S%z")
    rows = [[ts, method_name] + combo for combo in unique]
    pred_ws.append_rows(rows, value_input_option="RAW")
    print(f"[markov_mc] Wrote {len(rows)} predictions for {game} using method '{method_name}' (sims={sims}, temp={temp}, smooth={smooth}).")

def parse_targets_json(defaults: dict) -> dict:
    raw = os.environ.get("MARKOV_TARGETS_JSON", "").strip()
    if not raw:
        return defaults
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return {str(k): int(v) for k,v in d.items()}
    except Exception:
        pass
    return defaults

def main():
    # config
    method = os.environ.get("MARKOV_METHOD_NAME", "markov_mc")
    temp = env_float("MARKOV_TEMP", 0.8)
    smooth = env_float("MARKOV_SMOOTH", 1.0)
    sims = env_int("MARKOV_SIMS", 2000)
    unique_only = env_bool("MARKOV_UNIQUE", True)
    seed = os.environ.get("MARKOV_RANDOM_SEED")
    rng = random.Random(int(seed)) if seed and str(seed).strip()!="" else random.Random()

    # targets per game
    targets = parse_targets_json({"SuperCash": 10, "Badger 5": 5, "Megabucks": 10, "Powerball": 10})

    sh = open_sheet()

    # run only for games present in the sheet
    for game in list(targets.keys()):
        try:
            emit = int(targets[game])
            predict_for_game(sh, game, emit_count=emit, temp=temp, smooth=smooth, sims=sims, rng=rng, method_name=method, unique_only=unique_only)
        except Exception as e:
            print(f"[WARN] Skipping {game}: {e}")

if __name__ == "__main__":
    main()


# Fallback handling for empty dataset
def handle_empty_dataset(game, targets, generate_frequency_sampler_sets):
    print(f"[WARN] {game}: empty dataset for Markov; falling back to freq sampler.")
    return generate_frequency_sampler_sets(game, k=targets.get(game, 5))
