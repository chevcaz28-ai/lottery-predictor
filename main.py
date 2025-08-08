import os
import json
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import requests
from ml_model import train_and_predict_all

# ===== Google Sheets Auth =====
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(os.environ["GOOGLE_CREDS_JSON"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

SHEET_NAME = "Lottery Predictor New August 25"
sheet = client.open(SHEET_NAME)

# ===== Config =====
RAPID_API_KEY = os.environ["RAPID_API_KEY"]
API_URL = "https://lottery-results.p.rapidapi.com/games"
HEADERS = {
    "X-RapidAPI-Key": RAPID_API_KEY,
    "X-RapidAPI-Host": "lottery-results.p.rapidapi.com"
}

GAMES = {
    "Powerball": {"sheet": "Powerball_Results", "picks": 5, "extra": 1, "range": 69, "extra_range": 26},
    "Megabucks": {"sheet": "Megabucks_Results", "picks": 6, "extra": 0, "range": 49},
    "Super Cash": {"sheet": "SuperCash_Results", "picks": 6, "extra": 0, "range": 39},
    "Badger 5": {"sheet": "Badger5_Results", "picks": 5, "extra": 0, "range": 31}
}

# ===== Utility Functions =====
def get_worksheet(name):
    try:
        return sheet.worksheet(name)
    except gspread.WorksheetNotFound:
        sheet.add_worksheet(title=name, rows="5000", cols="20")
        return sheet.worksheet(name)

def append_to_sheet(ws, row):
    ws.append_row(row, value_input_option="USER_ENTERED")

def fetch_latest_results():
    response = requests.get(API_URL, headers=HEADERS)
    data = response.json()
    return data

def get_last_date(ws):
    vals = ws.col_values(1)
    if len(vals) > 1:
        return vals[-1]
    return None

def update_results(game, game_data):
    ws = get_worksheet(GAMES[game]["sheet"])
    last_date = get_last_date(ws)
    new_draw = False
    for entry in game_data["plays"][0]["draws"]:
        draw_date = entry["date"]
        nums = [n["value"] for n in entry["numbers"] if n["specialBall"] is None]
        extras = [n["value"] for n in entry["numbers"] if n["specialBall"] is not None]
        if draw_date != last_date:
            append_to_sheet(ws, [draw_date] + nums + extras)
            new_draw = True
    return new_draw

def update_tracker(game, predictions, method):
    ws = get_worksheet("prediction_tracker")
    today = datetime.now().strftime("%Y-%m-%d")
    for pred in predictions:
        row = [today, game, str(pred), method, "", "", ""]
        append_to_sheet(ws, row)

# ===== Main Logic =====
if __name__ == "__main__":
    data = fetch_latest_results()
    updated_games = []

    for game_id, game_data in data.items():
        name = game_data.get("name")
        if name in GAMES:
            if update_results(name, game_data):
                updated_games.append(name)

    if updated_games:
        all_predictions = train_and_predict_all(client, updated_games, GAMES)
        for game, pred_data in all_predictions.items():
            for method, preds in pred_data.items():
                update_tracker(game, preds, method)

    print("✅ Script executed successfully.")
