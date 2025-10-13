import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_predict_all(client, updated_games, GAMES):
    results = {}
    for game in updated_games:
        ws = client.open("Lottery Predictor New August 25").worksheet(GAMES[game]["sheet"])
        data = ws.get_all_values()[1:]
        df = pd.DataFrame(data)

        numbers = df.iloc[:, 1:GAMES[game]["picks"]+1].astype(int)
        all_nums = np.concatenate(numbers.values)

        # ===== Markov Chain =====
        markov_preds = markov_predict(all_nums, GAMES[game]["picks"], GAMES[game]["range"])

        # ===== Historical Probability =====
        hist_preds = historical_predict(all_nums, GAMES[game]["picks"], GAMES[game]["range"])

        # ===== ML Model =====
        ml_preds = ml_predict(numbers, GAMES[game]["picks"], GAMES[game]["range"])

        results[game] = {
            "Markov": [markov_preds],
            "Historical": [hist_preds],
            "ML": [ml_preds]
        }
    return results

def markov_predict(data, picks, num_range):
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(data)-1):
        counts[data[i]][data[i+1]] += 1
    top_nums = sorted(counts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:picks]
    return sorted([n for n, _ in top_nums])

def historical_predict(data, picks, num_range):
    values, counts = np.unique(data, return_counts=True)
    top_nums = values[np.argsort(-counts)][:picks]
    return sorted(top_nums)

def ml_predict(df, picks, num_range):
    X, y = [], []
    for i in range(len(df)-1):
        X.append(df.iloc[i].tolist())
        y.append(df.iloc[i+1].tolist()[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    probs = np.argsort(-clf.feature_importances_)[:picks] + 1
    return sorted(probs)
