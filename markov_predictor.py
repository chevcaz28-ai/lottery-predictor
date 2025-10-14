# markov_predictor.py — updated to use modern Sheets scopes & robust open logic
from __future__ import annotations
import os, json, time, random, math
from typing import List, Dict, Tuple
import pandas as pd

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# Local helper (same folder)
try:
    from sheets_util import open_spreadsheet, service_account_email
except Exception:
    # fallback inline if user didn't place sheets_util.py
    def _extract_id_from_url(url: str):
        import re
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
        return m.group(1) if m else None
    def open_spreadsheet():
        info = json.loads(os.environ["GOOGLE_CREDS_JSON"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        sheet_id = os.getenv("SHEET_ID","").strip()
        sheet_url = os.getenv("SHEET_URL","").strip()
        sheet_name = os.getenv("SHEET_NAME","").strip()
        if sheet_id:
            sh = gc.open_by_key(sheet_id)
        elif sheet_url:
            _id = _extract_id_from_url(sheet_url)
            if not _id:
                raise ValueError("SHEET_URL does not look like a Google Sheets URL")
            sh = gc.open_by_key(_id)
        elif sheet_name:
            sh = gc.open(sheet_name)
        else:
            raise RuntimeError("Set one of SHEET_ID, SHEET_URL, or SHEET_NAME.")
        return sh, gc, info.get("client_email","")
    def service_account_email():
        info = json.loads(os.environ["GOOGLE_CREDS_JSON"])
        return info.get("client_email","")

# -------------- Markov core (unchanged) --------------
def softmax(xs, temperature: float) -> List[float]:
    if temperature <= 0:
        temperature = 1e-6
    m = max(xs) if xs else 0.0
    exps = [math.exp((x - m)/temperature) for x in xs]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def build_transition_and_prior(history: List[List[int]], max_num: int, smooth: float):
    T = [[0.0]*(max_num+1) for _ in range(max_num+1)]
    freq = [0.0]*(max_num+1)
    for draw in history:
        for a in draw:
            freq[a] += 1.0
        for a in draw:
            for b in draw:
                if a != b:
                    T[a][b] += 1.0
    for i in range(1, max_num+1):
        row_sum = sum(T[i])
        if row_sum > 0:
            T[i] = [x/row_sum for x in T[i]]
    prior = [0.0] + [freq[j] + smooth for j in range(1, max_num+1)]
    return T, prior

def next_probs_from_last(last_set: List[int], T, prior, temperature: float) -> List[float]:
    max_num = len(prior)-1
    scores = [0.0]*(max_num+1)
    for j in range(1, max_num+1):
        scores[j] = prior[j]
    for i in last_set:
        row = T[i]
        for j in range(1, max_num+1):
            scores[j] += row[j]
    return [0.0] + softmax(scores[1:], temperature)

def pick_weighted_without_replacement(cands, weights, k, rng):
    chosen = []
    items = list(zip(cands, weights))
    total = sum(w for _,w in items) or 1.0
    items = [(c, (w/total if total>0 else 1.0/len(items))) for c,w in items]
    while len(chosen) < k and items:
        cs, ws = zip(*items)
        j = rng.choices(range(len(cs)), weights=ws, k=1)[0]
        chosen.append(cs[j])
        items.pop(j)
        total = sum(w for _,w in items) or 1.0
        if total > 0 and items:
            items = [(c, w/total) for c,w in items]
    return chosen

# -------------- Sheets helpers --------------
def ensure_predictions_ws(sh, game: str, n_numbers: int):
    title = f"{game}_Predictions"
    try:
        ws = sh.worksheet(title)
    except Exception:
        ws = sh.add_worksheet(title=title, rows=5000, cols=20)
        header = ["Timestamp", "Method"] + [f"N{i+1}" for i in range(n_numbers)]
        ws.update("A1", [header])
    vals = ws.get_all_values()
    if not vals:
        header = ["Timestamp", "Method"] + [f"N{i+1}" for i in range(n_numbers)]
        ws.update("A1", [header])
    return ws

def append_prediction(ws, method_name: str, numbers: List[int]):
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S+00:00")
    row = [ts, method_name] + numbers
    ws.append_row(row, value_input_option="RAW")

# -------------- Entry --------------
def run():
    rng = random.Random(int(os.getenv("MARKOV_RANDOM_SEED","123")))
    method_name = os.getenv("MARKOV_METHOD_NAME","markov_mc")
    temp = float(os.getenv("MARKOV_TEMP","0.8"))
    smooth = float(os.getenv("MARKOV_SMOOTH","1.0"))

    # Which games & counts to emit
    targets_json = os.getenv("MARKOV_TARGETS_JSON", '{"SuperCash":10,"Badger 5":5,"Megabucks":10,"Powerball":10}')
    targets = json.loads(targets_json)

    # Connect Sheets (now with modern scopes)
    try:
        sh, gc, sa_email = open_spreadsheet()
    except Exception as e:
        print(f"##[warning] Sheets connect failed: {e}")
        sh = None
        sa_email = service_account_email()

    if sa_email:
        print(f"[markov] Using service account: {sa_email}")

    # Example: fake last draw history per game (replace with your actual loader)
    game_meta = {
        "SuperCash": dict(max_num=39, draw_size=4),
        "Badger 5": dict(max_num=31, draw_size=5),
        "Megabucks": dict(max_num=49, draw_size=6),
        "Powerball": dict(max_num=69, draw_size=5, power_max=26),
    }

    total = 0
    for game, nrows in targets.items():
        meta = game_meta[game]
        max_num = meta["max_num"]
        draw_size = meta["draw_size"]
        # Dummy "history" — you should replace with your real past results load
        history = [sorted(rng.sample(range(1, max_num+1), draw_size)) for _ in range(100)]
        T, prior = build_transition_and_prior(history, max_num, smooth)

        ws = None
        if sh is not None:
            ws = ensure_predictions_ws(sh, game, draw_size)

        for _ in range(int(nrows)):
            last_set = history[-1]
            probs = next_probs_from_last(last_set, T, prior, temp)
            cands = list(range(1, max_num+1))
            weights = probs[1:]
            pick = sorted(pick_weighted_without_replacement(cands, weights, draw_size, rng))
            if game == "Powerball":
                pb = rng.randint(1, meta["power_max"])
                printable = f"[{game}] " + " ".join(map(str,pick)) + f" | PB {pb}"
            else:
                printable = f"[{game}] " + " ".join(map(str,pick))
            print(printable)
            total += 1

            if ws is not None:
                append_prediction(ws, method_name, pick)

    print(f"[{method_name}] Finished. Emitted total={total} predictions.")

if __name__ == "__main__":
    run()
