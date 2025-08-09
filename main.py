import os, json, datetime, itertools, warnings
import numpy as np
import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ========= CONFIG =========
SHEET_NAME = "Lottery Predictor New August 25"  # <-- make sure this matches your Sheet title
RAPID_API_KEY = os.environ["RAPID_API_KEY"]     # GitHub Secret
GOOGLE_CREDS_JSON = os.environ["GOOGLE_CREDS_JSON"]  # GitHub Secret (full JSON string)

GAMES = {
    "Powerball": {"tab":"Powerball_Results","picks":5,"maxn":69,"pb_maxn":26},
    "Megabucks": {"tab":"Megabucks_Results","picks":6,"maxn":49},
    "Super Cash": {"tab":"SuperCash_Results","picks":6,"maxn":39},
    "Badger 5": {"tab":"Badger5_Results","picks":5,"maxn":31},
}

# performance/learning knobs
FREQ_WINDOW = 100                   # recent draws to compute frequency
EVALUATION_WINDOW = 30              # recent rows considered in Best_Methods rollups
PERFORMANCE_THRESHOLD = 1.5         # avg matches to remain active
ALLOW_REACTIVATION = True
PREDICTIONS_PER_GAME = {"Powerball":5, "Megabucks":10, "Super Cash":10, "Badger 5":10}

# ========= SHEETS AUTH (google-auth) =========
creds_dict = json.loads(GOOGLE_CREDS_JSON)
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(creds)
ss = gc.open(SHEET_NAME)

def ws(name):
    try:
        return ss.worksheet(name)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=name, rows=5000, cols=20)

# ========= HELPERS =========
def read_df(name):
    vals = ws(name).get_all_records()
    return pd.DataFrame(vals)

def append_row(name, row):
    ws(name).append_row(row, value_input_option="USER_ENTERED")

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")

def log_run(updated_games, note=""):
    w = ws("Run_Log")
    if not w.get_all_values():
        w.append_row(["Timestamp","Updated Games","GPT Used","Notes"])
    append_row("Run_Log", [now_iso(), ", ".join(updated_games) if updated_games else "None", "No", note or "Scenario 3 auto run"])

def parse_results_row(game, draw):
    # draw["numbers"] contains entries with "value" and optional "specialBall"
    nums = [int(n["value"]) for n in draw["numbers"] if not n.get("specialBall")]
    extras = [int(n["value"]) for n in draw["numbers"] if n.get("specialBall")]
    if game=="Powerball":
        extras = extras[:1]
    return nums, extras

def last_date_in_tab(tab):
    data = ws(tab).get_all_values()
    if len(data) <= 1:
        return None
    return data[-1][0]  # first col Date

def fetch_latest_wi():
    """RapidAPI: WI results."""
    url = "https://lottery-results.p.rapidapi.com/games-by-state/us/wi"
    headers = {"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host":"lottery-results.p.rapidapi.com"}
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    return r.json()

def ensure_tracker_headers():
    w = ws("prediction_tracker")
    vals = w.get_all_values()
    if not vals:
        w.append_row(["Timestamp","Game","Prediction","Method","Win Count","Matches","Match Count"])

def ensure_best_methods_headers():
    w = ws("Best_Methods")
    vals = w.get_all_values()
    if not vals:
        w.append_row(["Game","Method","Total Predictions","Total Matches","Average Matches","Hit Rate (>=2)","Jackpot Wins","Last Evaluated"])

def ensure_active_methods_seed():
    w = ws("Active_Methods")
    vals = w.get_all_values()
    if not vals:
        w.append_row(["Game","Method","Active"])
        seeds = []
        for g in GAMES.keys():
            for m in ["Markovish","Historical","ML"]:
                seeds.append([g, m, "Yes"])
        w.append_rows(seeds, value_input_option="USER_ENTERED")

def read_active_methods():
    """Return dict: {game: set(methods active)}"""
    df = read_df("Active_Methods")
    if df.empty or "Game" not in df or "Method" not in df or "Active" not in df:
        return {g:set(["Markovish","Historical","ML"]) for g in GAMES.keys()}
    active = {}
    for g in GAMES.keys():
        sub = df[(df["Game"]==g) & (df["Active"].str.upper()=="YES")]
        active[g] = set(sub["Method"].tolist()) if not sub.empty else set(["Markovish","Historical","ML"])
    return active

# ========= EVALUATION =========
def evaluate_tracker_for_game(game):
    """Compare all unevaluated predictions to the latest draw for that game."""
    tracker = ws("prediction_tracker")
    rows = tracker.get_all_values()
    if len(rows) <= 1:
        return
    header, data = rows[0], rows[1:]
    # latest draw for game
    res_tab = GAMES[game]["tab"]
    res = read_df(res_tab)
    if res.empty:
        return
    last = res.iloc[-1].to_dict()
    main_cols = [c for c in res.columns if c.startswith("N")]
    last_main = set(int(last[c]) for c in main_cols if pd.notna(last[c]))

    for i, r in enumerate(data, start=2):
        row_game = r[1]; pred_str = r[2]; method = r[3]
        win = r[4]; matches_col = r[5]; match_count = r[6]
        if row_game != game:
            continue
        if str(match_count).strip() not in ("", "0"):
            continue
        # parse prediction numbers (ignore PB token for match scoring)
        parts = [p.strip() for p in pred_str.replace("[","").replace("]","").split(",") if p.strip()]
        if len(parts)==1 and "-" in parts[0]:
            parts = [x.strip() for x in parts[0].split("-")]
        pred_nums = []
        for p in parts:
            if p.upper().startswith("PB:"):
                continue
            try:
                pred_nums.append(int(p))
            except:
                pass
        sc = len(set(pred_nums) & last_main)
        tracker.update_cell(i, 5, 1 if sc == len(pred_nums) and len(pred_nums)>0 else 0)  # Win Count
        tracker.update_cell(i, 6, ",".join(map(str, sorted(set(pred_nums) & last_main))))  # Matches
        tracker.update_cell(i, 7, sc)  # Match Count

def update_best_methods():
    """Aggregate prediction_tracker into Best_Methods with rolling stats."""
    ensure_best_methods_headers()
    df = read_df("prediction_tracker")
    if df.empty:
        return
    df = df[pd.to_numeric(df["Match Count"], errors="coerce").notna()]
    if df.empty:
        return
    df["Match Count"] = df["Match Count"].astype(int)

    out_rows = []
    for (g, m), grp in df.groupby(["Game","Method"]):
        grp = grp.tail(EVALUATION_WINDOW)
        total_preds = len(grp)
        total_matches_sum = grp["Match Count"].sum()
        avg_matches = (total_matches_sum / total_preds) if total_preds else 0.0
        hit_rate = (grp["Match Count"] >= 2).mean() if total_preds else 0.0
        jackpot_wins = (grp["Win Count"].astype(str) == "1").sum() if "Win Count" in grp else 0
        last_eval = grp["Timestamp"].iloc[-1] if total_preds else ""
        out_rows.append([g, m, total_preds, total_matches_sum, round(avg_matches,3), round(hit_rate,3), int(jackpot_wins), last_eval])

    w = ws("Best_Methods")
    w.clear()
    w.append_row(["Game","Method","Total Predictions","Total Matches","Average Matches","Hit Rate (>=2)","Jackpot Wins","Last Evaluated"])
    if out_rows:
        w.append_rows(out_rows, value_input_option="USER_ENTERED")

def optimize_active_methods():
    """Flip Active_Methods Yes/No based on Best_Methods with thresholds."""
    bm = read_df("Best_Methods")
    if bm.empty:
        return
    am_ws = ws("Active_Methods")
    am = read_df("Active_Methods")
    if am.empty or "Game" not in am or "Method" not in am:
        ensure_active_methods_seed()
        am = read_df("Active_Methods")

    updated = []
    for _, row in bm.iterrows():
        g, m = row["Game"], row["Method"]
        avg = float(row["Average Matches"])
        total_preds = int(row["Total Predictions"])
        if total_preds >= EVALUATION_WINDOW:
            status = "Yes" if avg >= PERFORMANCE_THRESHOLD else "No"
        else:
            status = "Yes"  # collecting data
        if ALLOW_REACTIVATION and avg >= PERFORMANCE_THRESHOLD:
            status = "Yes"
        updated.append((g,m,status))

    store = {}
    for _, r in am.iterrows():
        store[(r["Game"], r["Method"])] = r["Active"]
    for g,m,s in updated:
        store[(g,m)] = s
    am_ws.clear()
    am_ws.append_row(["Game","Method","Active"])
    rows = [[g,m,store.get((g,m),"Yes")] for g in GAMES.keys() for m in ["Markovish","Historical","ML"]]
    am_ws.append_rows(rows, value_input_option="USER_ENTERED")

# ========= FEATURES + MODELS =========
def history_numeric(game):
    tab = GAMES[game]["tab"]
    df = read_df(tab)
    if df.empty:
        return pd.DataFrame()
    num_cols = [c for c in df.columns if c.startswith("N")] + (["PB"] if "PB" in df.columns else [])
    out = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    return out

def build_features(history_nums, maxn):
    if history_nums.empty:
        return pd.DataFrame({"num":range(1,maxn+1),"freq":0,"recency":999,"co_last":0.0})
    last_k = history_nums.tail(FREQ_WINDOW)
    vals = last_k.to_numpy().ravel()
    vals = vals[~np.isnan(vals)].astype(int)
    freq = pd.Series(vals).value_counts().reindex(range(1,maxn+1), fill_value=0)

    rec = {n:999 for n in range(1,maxn+1)}
    for i in range(history_nums.shape[0]-1, -1, -1):
        draw = set(history_nums.iloc[i].dropna().astype(int).tolist())
        for n in draw:
            if rec[n]==999:
                rec[n] = history_nums.shape[0] - i
        if all(rec[n]!=999 for n in range(1,maxn+1)):
            break

    last_draw = set(history_nums.iloc[-1].dropna().astype(int).tolist())
    co = {n:0 for n in range(1,maxn+1)}
    for _, row in last_k.iterrows():
        rset = set(row.dropna().astype(int).tolist())
        for n in rset:
            co[n] += len(rset & last_draw)
    co_s = pd.Series(co) / max(1, last_k.shape[0])

    feat = pd.DataFrame({
        "num": range(1,maxn+1),
        "freq": freq.values,
        "recency": pd.Series(rec).values,
        "co_last": co_s.values
    })
    return feat

def train_per_number_rf(history_nums, maxn):
    if history_nums.shape[0] < 30:
        return None
    rows = []
    Y = []
    for t in range(10, history_nums.shape[0]-1):
        sub = history_nums.iloc[:t+1]
        nxt = set(history_nums.iloc[t+1].dropna().astype(int).tolist())
        feat = build_features(sub, maxn)[["freq","recency","co_last"]].values
        rows.append(feat)
        Y.append(np.array([1 if n in nxt else 0 for n in range(1,maxn+1)]))
    if not rows:
        return None
    X = np.vstack(rows)
    Y = np.hstack(Y)
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=300, random_state=42))])
    pipe.fit(X, Y)
    return pipe

def scores_from_model(model, history_nums, maxn):
    feat = build_features(history_nums, maxn)[["freq","recency","co_last"]].values
    if model is None:
        base = feat[:,0] + (100 - np.clip(feat[:,1],0,100)) + feat[:,2]
        p = np.maximum(base, 0); p = p / (p.sum() if p.sum() else 1)
        return np.arange(1,maxn+1), p
    # model may not have predict_proba for multilabel-like target; fallback to feature-based score
    try:
        proba = model.predict_proba(feat)[:,1]
    except Exception:
        proba = np.maximum(feat[:,0] + (100 - np.clip(feat[:,1],0,100)) + feat[:,2], 0.0)
    p = proba / (proba.sum() if proba.sum() else 1)
    return np.arange(1,maxn+1), p

def pick_sets_from_probs(nums, probs, k, count):
    res = []
    for _ in range(count):
        choice = np.random.choice(nums, size=k, replace=False, p=probs)
        res.append(sorted(map(int, choice)))
    return res

def historical_sets(history_nums, maxn, k, count):
    vals = history_nums.tail(FREQ_WINDOW).to_numpy().ravel()
    vals = vals[~np.isnan(vals)].astype(int)
    freq = pd.Series(vals).value_counts().reindex(range(1,maxn+1), fill_value=0).values.astype(float)
    freq += 1e-3
    probs = freq / freq.sum()
    return pick_sets_from_probs(np.arange(1,maxn+1), probs, k, count)

def markovish_sets(history_nums, maxn, k, count):
    feat = build_features(history_nums, maxn)
    score = feat["freq"].values + 2.0*feat["co_last"].values + 0.25*(100 - np.clip(feat["recency"].values,0,100))
    score = np.maximum(score, 0.0)
    probs = score / (score.sum() if score.sum() else 1)
    return pick_sets_from_probs(np.arange(1,maxn+1), probs, k, count)

def push_predictions(game, method, sets, history_nums):
    ensure_tracker_headers()
    recent = history_nums.tail(10) if not history_nums.empty else pd.DataFrame()
    for s in sets:
        append_row("prediction_tracker", [now_iso(), game, "-".join(map(str,s)), method, "", "", ""])

def add_pb(sets, pb_series, pb_maxn):
    if pb_series is not None and not pb_series.dropna().empty:
        pb_vals = pb_series.tail(FREQ_WINDOW).dropna().astype(int)
        counts = pd.Series(pb_vals).value_counts().reindex(range(1,pb_maxn+1), fill_value=0).values.astype(float)
        counts += 1e-3
        probs = counts / counts.sum()
    else:
        probs = np.ones(pb_maxn)/pb_maxn
    out=[]
    for s in sets:
        pb = int(np.random.choice(np.arange(1,pb_maxn+1), p=probs))
        out.append(s + [f"PB:{pb}"])
    return out

# ========= MAIN =========
def main():
    ensure_tracker_headers()
    ensure_best_methods_headers()
    ensure_active_methods_seed()

    active_by_game = read_active_methods()
    data = fetch_latest_wi()

    updated_games = []

    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name not in GAMES:
            continue
        tab = GAMES[name]["tab"]
        last_date = last_date_in_tab(tab)
        draw = entry["plays"][0]["draws"][0]
        draw_date = draw["date"]

        if draw_date == last_date:
            continue  # no update

        nums, extras = parse_results_row(name, draw)
        row = [draw_date] + nums
        if name=="Powerball":
            row += [extras[0] if extras else ""]
        elif name=="Super Cash":
            doubler = "N"
            for n in draw["numbers"]:
                if n.get("specialBall") and n["specialBall"].get("name")=="Doubler":
                    doubler = n["value"]
            row += [doubler]
        append_row(tab, row)
        updated_games.append(name)

    # Evaluate predictions for updated games
    for g in updated_games:
        evaluate_tracker_for_game(g)

    # Update method leaderboard and optimize actives
    update_best_methods()
    optimize_active_methods()
    active_by_game = read_active_methods()  # refresh

    # Generate predictions only for updated games
    for g in updated_games:
        cfg = GAMES[g]
        hist = history_numeric(g)
        if hist.empty:
            continue
        k = cfg["picks"]
        count = PREDICTIONS_PER_GAME[g]
        actives = active_by_game.get(g, set(["Markovish","Historical","ML"]))
        out_sets = []

        # ML
        if "ML" in actives:
            model = train_per_number_rf(hist[[c for c in hist.columns if c.startswith("N")]], cfg["maxn"])
            nums, probs = scores_from_model(model, hist[[c for c in hist.columns if c.startswith("N")]], cfg["maxn"])
            ml_sets = pick_sets_from_probs(nums, probs, k, count)
            out_sets.append(("ML", ml_sets))

        # Historical
        if "Historical" in actives:
            hist_sets = historical_sets(hist[[c for c in hist.columns if c.startswith("N")]], cfg["maxn"], k, count)
            out_sets.append(("Historical", hist_sets))

        # Markovish
        if "Markovish" in actives:
            mk_sets = markovish_sets(hist[[c for c in hist.columns if c.startswith("N")]], cfg["maxn"], k, count)
            out_sets.append(("Markovish", mk_sets))

        # attach PB for Powerball
        for method, sets in out_sets:
            final_sets = sets
            if g=="Powerball":
                pb_series = hist["PB"] if "PB" in hist.columns else None
                final_sets = add_pb(sets, pb_series, cfg["pb_maxn"])
            push_predictions(g, method, final_sets, hist)

    log_run(updated_games, note="Updated results, evaluated, predicted (active methods only)")
    print("✅ Script executed successfully.")

if __name__ == "__main__":
    main()
