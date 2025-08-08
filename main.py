"""
Lottery Full Auto + Pro ML
Scenario 3 Final Script
"""

import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# ===== CONFIG =====
RAPID_API_KEY = os.getenv("RAPID_API_KEY")  # Set in GitHub Secrets
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")  # Set in GitHub Secrets

SHEET_NAME = "Lottery Predictor New August 25"  # Change to match your sheet
MODEL_FILE = "ml_model.pkl"

GAMES_CONFIG = {
    "Powerball": {"range": 69, "picks": 5, "extra_range": 26, "extra_picks": 1},
    "Megabucks": {"range": 49, "picks": 6},
    "Super Cash": {"range": 39, "picks": 6},
    "Badger 5": {"range": 31, "picks": 5},
}

# ===== GOOGLE SHEETS AUTH =====
creds_dict = json.loads(GOOGLE_CREDS_JSON)
creds = Credentials.from_service_account_info(creds_dict, scopes=[
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
])
client = gspread.authorize(creds)
sheet = client.open(SHEET_NAME)

# ===== HELPER: SHEETS =====
def read_sheet(tab_name):
    df = pd.DataFrame(sheet.worksheet(tab_name).get_all_records())
    return df

def write_sheet(tab_name, df):
    ws = sheet.worksheet(tab_name)
    ws.clear()
    ws.update([df.columns.values.tolist()] + df.values.tolist())

# ===== FETCH RESULTS =====
def fetch_latest_results():
    url = "https://lotteryapi.p.rapidapi.com/api/results"
    headers = {
        "x-rapidapi-host": "lotteryapi.p.rapidapi.com",
        "x-rapidapi-key": RAPID_API_KEY
    }
    params = {"state": "wi"}
    r = requests.get(url, headers=headers, params=params)
    data = r.json()
    return data

# ===== ML FUNCTIONS =====
def train_ml_model(history_df):
    """Train ML model based on historical prediction performance."""
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(history_df["Prediction"].apply(lambda x: tuple(map(int, x.split("-")))))
    y = history_df["Matches"]  # Match count is target
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, mlb

def predict_with_ml(model, mlb, game_cfg, top_candidates):
    """Use ML to score candidate predictions."""
    binarized = mlb.transform(top_candidates)
    scores = model.predict_proba(binarized)[:, 1] if len(model.classes_) > 1 else np.zeros(len(top_candidates))
    ranked = [x for _, x in sorted(zip(scores, top_candidates), reverse=True)]
    return ranked[:5]

# ===== PREDICTION METHODS =====
def markov_prediction(history, game_cfg):
    numbers = history["Numbers"].explode().value_counts().index.tolist()
    return sorted(random.sample(numbers[:15], game_cfg["picks"]))

def freq_prediction(history, game_cfg):
    counts = history["Numbers"].explode().value_counts()
    return sorted(random.sample(counts.head(15).index.tolist(), game_cfg["picks"]))

def random_prediction(game_cfg):
    return sorted(random.sample(range(1, game_cfg["range"] + 1), game_cfg["picks"]))

# ===== MAIN =====
def main():
    # Load past prediction tracker
    tracker_df = read_sheet("prediction_tracker")

    # Load or train ML model
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model, mlb = pickle.load(f)
    else:
        model, mlb = train_ml_model(tracker_df)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, mlb), f)

    # Fetch latest results
    data = fetch_latest_results()

    for _, game_cfg in GAMES_CONFIG.items():
        pass  # Ensure no unused variable errors

    updated_tracker = tracker_df.copy()

    for entry in data.values():
        game_name = entry["name"]
        if game_name not in GAMES_CONFIG:
            continue

        game_cfg = GAMES_CONFIG[game_name]
        draws = entry["plays"][0]["draws"]
        latest_draw = draws[0]
        draw_date = latest_draw["date"]
        numbers = [int(n["value"]) for n in latest_draw["numbers"] if not n.get("specialBall")]
        extra_num = [int(n["value"]) for n in latest_draw["numbers"] if n.get("specialBall") and game_cfg.get("extra_range")]

        # Update history
        history_df = read_sheet(game_name)
        if draw_date not in history_df["Date"].values:
            new_row = {"Date": draw_date, "Numbers": numbers}
            if extra_num:
                new_row["Extra"] = extra_num[0]
            history_df = pd.concat([pd.DataFrame([new_row]), history_df], ignore_index=True)
            write_sheet(game_name, history_df)

        # Generate predictions
        markov = markov_prediction(history_df, game_cfg)
        freq = freq_prediction(history_df, game_cfg)
        rand = random_prediction(game_cfg)
        candidates = [markov, freq, rand]

        ml_preds = predict_with_ml(model, mlb, game_cfg, candidates)
        final_preds = ml_preds if ml_preds else candidates

        # Append predictions to tracker
        for pred in final_preds:
            updated_tracker = pd.concat([updated_tracker, pd.DataFrame([{
                "Date": draw_date,
                "Game": game_name,
                "Prediction": "-".join(map(str, pred)),
                "Method": "ML Ensemble",
                "Win Count": 0,
                "Matches": 0,
                "Match Count": 0
            }])], ignore_index=True)

    write_sheet("prediction_tracker", updated_tracker)

    # Retrain model
    model, mlb = train_ml_model(updated_tracker)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, mlb), f)

    print("✅ Script executed successfully.")

if __name__ == "__main__":
    main()
