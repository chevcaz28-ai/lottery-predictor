import json
import os
import pandas as pd
import numpy as np
import gspread
import requests
import datetime
from oauth2client.service_account import ServiceAccountCredentials

# Load credentials.json from GitHub Secrets
creds_json = os.environ.get("GOOGLE_CREDS_JSON")
with open("credentials.json", "w") as f:
    f.write(creds_json)

# Config
SHEET_NAME = "Lottery Predictor New August 25"
OPENAI_KEY = os.environ.get("OPENAI_KEY")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")

games_config = {
    "Badger5": {"picks": 5, "range": 31, "predictions": 10},
    "SuperCash": {"picks": 6, "range": 39, "predictions": 10},
    "Megabucks": {"picks": 6, "range": 49, "predictions": 10},
    "Powerball": {"picks": 5, "range": 69, "pb_range": 26, "predictions": 5},
}

LAST_DRAWS = 10
FREQ_WINDOW = 50

# Authenticate Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

def log_run(updated_games, gpt_used):
    try:
        ws = client.open(SHEET_NAME).worksheet("Run_Log")
    except:
        ws = client.open(SHEET_NAME).add_worksheet("Run_Log", rows=1000, cols=4)
        ws.append_row(["Timestamp", "Updated Games", "GPT Used", "Prediction Counts"])
    counts = {g: games_config[g]["predictions"] for g in updated_games}
    ws.append_row([
        str(datetime.datetime.now()),
        ", ".join(updated_games) if updated_games else "None",
        "Yes" if gpt_used else "No",
        json.dumps(counts)
    ])

def get_game_history(game):
    try:
        ws = client.open(SHEET_NAME).worksheet(game)
    except:
        return pd.DataFrame()
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return pd.DataFrame()
    num_cols = [c for c in df.columns if c.startswith("N") or c == "PB"]
    return df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()

def build_frequency_table(draws, value_range):
    values = draws.stack().astype(int)
    return values.value_counts().reindex(range(1, value_range+1), fill_value=0)

def calc_similarity(prediction, recent_draws):
    nums = [int(x) for x in prediction if isinstance(x, (int, np.integer))]
    return max((len(set(nums) & set(draw)) for draw in recent_draws), default=0)

def push_to_tracker(predictions, method, source, game_histories):
    ws = client.open(SHEET_NAME).worksheet("Prediction_Tracker")
    next_id = len(ws.get_all_values())
    for game, sets in predictions.items():
        recent_draws = game_histories.get(game, [])
        for s in sets:
            s_clean = [int(x) if isinstance(x,(np.integer,int)) else x for x in s]
            similarity = calc_similarity(s_clean, recent_draws)
            ws.append_row([
                next_id+1, game, str(datetime.datetime.now()), "-".join(map(str,s_clean)),
                method, "", source, method, "", "", "", "Pending", "", "", "",
                "Yes", method, similarity, ""
            ])
            next_id += 1

def fetch_and_update_results():
    url = "https://lottery-results.p.rapidapi.com/games-by-state/us/wi"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "lottery-results.p.rapidapi.com"}
    data = requests.get(url, headers=headers).json()
    updated_games = []

    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("name") not in ["Powerball", "Megabucks", "Super Cash", "Badger 5"]:
            continue

        draw = entry["plays"][0]["draws"][0]
        draw_date = draw["date"]
        nums = [n["value"] for n in draw["numbers"]]
        special = next((n["value"] for n in draw["numbers"] if n.get("specialBall")), None)

        if entry["name"] == "Powerball":
            row = [draw_date] + [n for n in nums if n.isdigit()][:5] + [special or ""]
            tab = "Powerball"
        elif entry["name"] == "Super Cash":
            row = [draw_date] + [n for n in nums if n.isdigit()][:6]
            doubler = next((n["value"] for n in draw["numbers"] if n["value"] in ["Y","N"]), "N")
            row.append(doubler)
            tab = "SuperCash"
        elif entry["name"] == "Megabucks":
            row = [draw_date] + [n for n in nums if n.isdigit()][:6]
            tab = "Megabucks"
        else:
            row = [draw_date] + [n for n in nums if n.isdigit()][:5]
            tab = "Badger5"

        ws = client.open(SHEET_NAME).worksheet(tab)
        all_dates = [r[0] for r in ws.get_all_values()[1:]]
        if draw_date not in all_dates:
            ws.append_row(row)
            updated_games.append(tab)

    return updated_games

if __name__ == "__main__":
    updated_games = fetch_and_update_results()

    if not updated_games:
        log_run([], False)
    else:
        predictions_markov = {}
        predictions_gpt = {}
        game_histories = {}
        prompt_context = ""

        for game in updated_games:
            config = games_config[game]
            draws = get_game_history(game)
            last_draws = draws.tail(LAST_DRAWS).values.tolist()
            freq_table = build_frequency_table(draws.tail(FREQ_WINDOW), config["range"])
            game_histories[game] = last_draws

            markov_sets = [sorted(np.random.choice(range(1, config["range"]+1), config["picks"], replace=False))
                           for _ in range(config["predictions"])]
            predictions_markov[game] = markov_sets

            prompt_context += f"\nGame: {game}\nRecent {LAST_DRAWS} draws:\n{last_draws}\n"
            prompt_context += f"Number frequencies (last {FREQ_WINDOW} draws):\n{freq_table.to_dict()}\n"

        prompt = f"""
        You are predicting lottery numbers using historical patterns.

        Context:
        {prompt_context}

        Generate predictions only for these games: {updated_games}

        Rules:
        - Generate complete sets that resemble historical patterns.
        - Avoid repeating the exact last 3 draws for each game.
        - Numbers must be sorted and unique.
        - Return JSON ONLY like:
        {{
         "Badger5": [[1,2,3,4,5], ...],
         "SuperCash": [[...6 numbers...], ...],
         "Megabucks": [[...6 numbers...], ...],
         "Powerball": [[1,2,3,4,5], ...]
        }}
        """

        headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                             json={"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}]})

        try:
            gpt_output = json.loads(resp.json()["choices"][0]["message"]["content"])
            predictions_gpt = gpt_output
            gpt_used = True
        except:
            predictions_gpt = {}
            gpt_used = False

        if "Powerball" in predictions_gpt:
            draws = get_game_history("Powerball")
            if draws.empty or "PB" not in draws.columns:
                pb_freq = pd.Series(range(1, games_config["Powerball"]["pb_range"]+1))
            else:
                pb_values = draws["PB"].values[-FREQ_WINDOW:]
                pb_freq = pd.Series(pb_values).value_counts().sort_values(ascending=False)

            updated_pb = []
            for s in predictions_gpt["Powerball"]:
                pb_num = int(np.random.choice(pb_freq.index[:10]))
                updated_pb.append(s + [f"PB:{pb_num}"])
            predictions_gpt["Powerball"] = updated_pb

        push_to_tracker(predictions_markov, "Markov", "Markov Only", game_histories)
        if predictions_gpt:
            push_to_tracker(predictions_gpt, "GPT-Hybrid", "GPT Hybrid", game_histories)

        log_run(updated_games, gpt_used)

    print("✅ Script executed successfully.")

