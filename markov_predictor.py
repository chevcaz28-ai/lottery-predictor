#!/usr/bin/env python3
"""
markov_predictor.py

Purpose
- Generate Markov-style lottery predictions (position-wise transition model) per game.
- Run a rolling backtest and write results + summary to Google Sheets.

Design goals
- Self-contained and safe: if anything fails, prints a clear error and exits non-zero.
- Does NOT modify main prediction logic; it only reads *_Results tabs and writes its own tabs.
- Always emits backtest info to logs so you can verify in Actions even if Sheet write fails.

Expected inputs (env)
- GOOGLE_CREDS_JSON: Google service account JSON string
- SHEET_NAME: spreadsheet name (or set SHEET_ID to open by id)
- SHEET_ID: (optional) spreadsheet id
- LOCAL_TZ: timezone name (default America/Chicago)
- MARKOV_TARGETS_JSON: JSON object like {"Powerball":10,"Megabucks":10,"SuperCash":10,"Badger 5":5}
- MARKOV_TEMP: float temperature for sampling (default 1.0)
- MARKOV_SMOOTH: float Laplace smoothing (default 1.0)
- MARKOV_RANDOM_SEED: int (default 123)
- MARKOV_BACKTEST_DRAWS: how many most-recent draws to backtest (default 200)
- MARKOV_LOOKBACK_MIN: minimum prior draws required to start backtest (default 200)

Writes (tabs)
- Markov_Picks
- Markov_Backtest
"""

from __future__ import annotations

import os
import sys
import json
import re
import math
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import pytz
import gspread
from google.oauth2.service_account import Credentials


# -----------------------------
# Config / helpers
# -----------------------------

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing required env var: {name}")
        return default
    return v

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        raise RuntimeError(f"Env var {name} must be float, got: {v!r}")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        raise RuntimeError(f"Env var {name} must be int, got: {v!r}")

def _now_iso(tzname: str) -> str:
    tz = pytz.timezone(tzname)
    return datetime.now(tz).isoformat()

def _parse_json_maybe(masked: str) -> Dict:
    """
    MARKOV_TARGETS_JSON may be masked in logs, but the actual env value should be valid JSON.
    Support either:
      - '{"Powerball":10,...}'
      - '***"Powerball":10,...***' (masked fragments) -> will fail; we raise clearly
    """
    s = (masked or "").strip()
    if not s:
        return {}
    if s.startswith("***") and s.endswith("***"):
        # In GitHub logs, secrets are masked, but env at runtime is NOT masked.
        # If someone literally stored ***...***, we can't parse it.
        raise RuntimeError("MARKOV_TARGETS_JSON appears masked (starts/ends with ***). "
                           "Check that the secret value in GitHub is valid JSON (without ***).")
    # Allow single quotes accidentally
    if s.startswith("{'") or s.startswith("{\""):
        pass
    try:
        return json.loads(s)
    except Exception as e:
        raise RuntimeError(f"MARKOV_TARGETS_JSON is not valid JSON: {e}. Value starts with: {s[:50]!r}")

def _gspread_client_from_service_account_json(creds_json: str) -> gspread.Client:
    info = json.loads(creds_json)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def _open_sheet(gc: gspread.Client, sheet_name: str) -> gspread.Spreadsheet:
    sheet_id = os.getenv("SHEET_ID", "").strip()
    if sheet_id:
        return gc.open_by_key(sheet_id)
    return gc.open(sheet_name)

def _worksheet_get_df(ws: gspread.Worksheet) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    return df

def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("date", "drawdate", "draw_date"):
            return c
    # fallback: first col with 'date' in name
    for c in df.columns:
        if "date" in c.strip().lower():
            return c
    return None

def _coerce_date_series(s: pd.Series) -> pd.Series:
    # Try common formats; coerce errors to NaT
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def _infer_number_cols(df: pd.DataFrame) -> List[str]:
    """
    Try to identify draw number columns.
    Strategy:
      - prefer columns whose header is numeric-ish like 'n1','num1','ball1', 'white1'...
      - else, take columns that are mostly digits and exclude obvious non-numbers
    """
    cols = list(df.columns)
    preferred = []
    for c in cols:
        cl = c.strip().lower()
        if any(k in cl for k in ["n1","n2","n3","n4","n5","n6","num","ball","white","red","pb","mb","bonus"]):
            preferred.append(c)
    # If preferred is too wide, we'll filter by content digits
    def digit_ratio(col: str) -> float:
        vals = df[col].astype(str).str.strip()
        nonempty = vals[vals != ""]
        if len(nonempty) == 0:
            return 0.0
        isdig = nonempty.str.fullmatch(r"\d+").fillna(False)
        return float(isdig.mean())
    numericish = [c for c in cols if digit_ratio(c) >= 0.8]
    # Remove date col and obvious text cols
    date_c = _find_date_col(df)
    bad = set()
    if date_c:
        bad.add(date_c)
    for c in cols:
        cl = c.strip().lower()
        if any(k in cl for k in ["game","state","jackpot","multiplier","draw","weekday"]):
            bad.add(c)
    numericish = [c for c in numericish if c not in bad]
    # If preferred intersects numericish, use that intersection in original order
    use = [c for c in cols if c in numericish and c in preferred]
    if len(use) >= 3:
        return use
    # else use numericish in original order
    return [c for c in cols if c in numericish]

@dataclass
class GameSpec:
    name: str
    results_tab: str
    main_count: int
    main_min: int
    main_max: int
    bonus_count: int = 0
    bonus_min: int = 0
    bonus_max: int = 0

# Based on your sheet tab names in the log
GAME_SPECS: Dict[str, GameSpec] = {
    "Powerball": GameSpec("Powerball", "Powerball_Results", 5, 1, 69, bonus_count=1, bonus_min=1, bonus_max=26),
    "Megabucks": GameSpec("Megabucks", "Megabucks_Results", 6, 1, 49),
    "SuperCash": GameSpec("SuperCash", "SuperCash_Results", 6, 1, 45),
    "Badger 5": GameSpec("Badger 5", "Badger5_Results", 5, 1, 31),
}

# -----------------------------
# Markov model (position-wise)
# -----------------------------

class PositionMarkov:
    """
    For each position j, learn P(next_num | prev_num) as a transition matrix.
    We also keep marginals to support unseen prev numbers.

    This is not a true Markov chain over the full vector state; it's a pragmatic,
    stable model that still captures "carryover" tendencies per position.
    """
    def __init__(self, min_num: int, max_num: int, smooth: float = 1.0):
        self.min_num = min_num
        self.max_num = max_num
        self.K = max_num - min_num + 1
        self.smooth = float(smooth)
        # counts[prev, next]
        self.counts = np.zeros((self.K, self.K), dtype=np.float64)
        self.next_marg = np.zeros((self.K,), dtype=np.float64)

    def _idx(self, n: int) -> int:
        return int(n) - self.min_num

    def fit_pairs(self, prev: List[int], nxt: List[int]) -> None:
        for a, b in zip(prev, nxt):
            if not (self.min_num <= a <= self.max_num and self.min_num <= b <= self.max_num):
                continue
            ia = self._idx(a)
            ib = self._idx(b)
            self.counts[ia, ib] += 1.0
            self.next_marg[ib] += 1.0

    def probs_next(self, prev_n: int, temp: float = 1.0) -> np.ndarray:
        # Laplace smoothing
        if self.min_num <= prev_n <= self.max_num:
            row = self.counts[self._idx(prev_n), :] + self.smooth
        else:
            row = self.next_marg + self.smooth  # fallback to marginal if prev unseen
        p = row / row.sum()
        # temperature: p_i^(1/temp) then renormalize (temp>1 -> flatter)
        t = max(float(temp), 1e-6)
        if abs(t - 1.0) > 1e-9:
            p = np.power(p, 1.0 / t)
            p = p / p.sum()
        return p

def _sample_unique(probs: np.ndarray, n: int, rng: np.random.Generator) -> List[int]:
    """
    Sample n unique indices without replacement, biased by probs.
    """
    probs = np.array(probs, dtype=np.float64)
    probs = probs / probs.sum()
    chosen = []
    avail = np.arange(len(probs))
    p = probs.copy()
    for _ in range(n):
        if len(avail) == 0:
            break
        idx = rng.choice(avail, p=p)
        chosen.append(int(idx))
        # remove idx
        mask = avail != idx
        avail = avail[mask]
        p = p[mask]
        if p.sum() <= 0:
            p = np.ones_like(p, dtype=np.float64) / len(p)
        else:
            p = p / p.sum()
    return chosen

def _normalize_draw(nums: List[int], spec: GameSpec) -> List[int]:
    # sort mains; bonus stays at end
    mains = sorted(nums[:spec.main_count])
    if spec.bonus_count:
        return mains + nums[spec.main_count:spec.main_count + spec.bonus_count]
    return mains

def _format_pick(nums: List[int], spec: GameSpec) -> str:
    mains = nums[:spec.main_count]
    if spec.bonus_count:
        bonus = nums[spec.main_count]
        return f"{'-'.join(str(x) for x in mains)} | {bonus}"
    return "-".join(str(x) for x in mains)

def build_models(draws: List[List[int]], spec: GameSpec, smooth: float) -> Tuple[List[PositionMarkov], Optional[PositionMarkov]]:
    """
    draws is list of normalized draws (sorted mains, bonus last if any).
    Return (main_models, bonus_model)
    """
    main_models = [PositionMarkov(spec.main_min, spec.main_max, smooth=smooth) for _ in range(spec.main_count)]
    bonus_model = PositionMarkov(spec.bonus_min, spec.bonus_max, smooth=smooth) if spec.bonus_count else None

    for i in range(len(draws) - 1):
        prev = draws[i]
        nxt = draws[i + 1]
        # main positions
        for j in range(spec.main_count):
            main_models[j].fit_pairs([prev[j]], [nxt[j]])
        # bonus
        if spec.bonus_count and bonus_model is not None:
            bonus_model.fit_pairs([prev[spec.main_count]], [nxt[spec.main_count]])
    return main_models, bonus_model

def generate_picks(draws: List[List[int]], spec: GameSpec, k: int, temp: float, smooth: float, seed: int) -> List[List[int]]:
    if len(draws) < 3:
        return []
    rng = np.random.default_rng(seed)
    main_models, bonus_model = build_models(draws, spec, smooth=smooth)
    last = draws[-1]
    picks: List[List[int]] = []
    # We'll create slight diversity by varying per-pick seed offset
    for i in range(k):
        mains = []
        for j, model in enumerate(main_models):
            p = model.probs_next(last[j], temp=temp)
            # sample one number from distribution
            idx = rng.choice(np.arange(model.K), p=p)
            n = spec.main_min + int(idx)
            mains.append(n)
        # enforce unique mains (lottery mains are unique)
        # If duplicates arise, resample using a pooled distribution:
        if len(set(mains)) != len(mains):
            # pooled probs: average of position probs
            pooled = np.zeros((spec.main_max - spec.main_min + 1,), dtype=np.float64)
            for j, model in enumerate(main_models):
                pooled += model.probs_next(last[j], temp=temp)
            pooled = pooled / pooled.sum()
            # sample unique mains
            idxs = _sample_unique(pooled, spec.main_count, rng)
            mains = [spec.main_min + ii for ii in idxs]
        mains = sorted(mains)

        out = mains
        if spec.bonus_count and bonus_model is not None:
            pb_probs = bonus_model.probs_next(last[spec.main_count], temp=temp)
            pb_idx = rng.choice(np.arange(bonus_model.K), p=pb_probs)
            pb = spec.bonus_min + int(pb_idx)
            out = mains + [pb]
        picks.append(out)
    return picks

def _match_score(actual: List[int], pred: List[int], spec: GameSpec) -> Tuple[int, int]:
    """
    returns (main_matches, bonus_match)
    """
    a_main = set(actual[:spec.main_count])
    p_main = set(pred[:spec.main_count])
    main = len(a_main & p_main)
    bonus = 0
    if spec.bonus_count:
        bonus = 1 if actual[spec.main_count] == pred[spec.main_count] else 0
    return main, bonus

def backtest(draws: List[List[int]], spec: GameSpec, k: int, temp: float, smooth: float, seed: int,
             lookback_min: int, backtest_draws: int) -> pd.DataFrame:
    """
    Rolling backtest over the last `backtest_draws` draws.
    For each t, train on draws[:t], predict draws[t], score.
    """
    n = len(draws)
    if n <= lookback_min + 1:
        return pd.DataFrame()

    start = max(lookback_min, n - backtest_draws)
    rows = []
    rng_seed = seed

    for t in range(start, n - 1):
        train = draws[:t]
        actual = draws[t]
        preds = generate_picks(train, spec, k=k, temp=temp, smooth=smooth, seed=rng_seed)
        rng_seed += 1
        if not preds:
            continue
        scores = [_match_score(actual, p, spec) for p in preds]
        main_scores = [s[0] for s in scores]
        bonus_scores = [s[1] for s in scores]
        rows.append({
            "index": t,
            "actual": _format_pick(actual, spec),
            "pred_count": len(preds),
            "best_main": int(max(main_scores)),
            "avg_main": float(np.mean(main_scores)),
            "any_3plus_main": int(any(m >= 3 for m in main_scores)),
            "any_4plus_main": int(any(m >= 4 for m in main_scores)),
            "any_5plus_main": int(any(m >= 5 for m in main_scores)),
            "best_bonus": int(max(bonus_scores)) if spec.bonus_count else "",
            "any_bonus": int(any(b == 1 for b in bonus_scores)) if spec.bonus_count else "",
        })
    return pd.DataFrame(rows)

def _ensure_ws(ss: gspread.Spreadsheet, title: str, rows: int = 2000, cols: int = 26) -> gspread.Worksheet:
    try:
        return ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def _open_ws_fuzzy(ss, names):
    """Try multiple worksheet names; return first that exists."""
    last = None
    for n in names:
        try:
            return ss.worksheet(n)
        except Exception as e:
            last = e
            continue
    raise last if last else Exception("Worksheet not found")


def _row_from_header(header, values_by_key):
    """Build a row array aligned to header. Unmapped columns get ''."""
    hnorm = [str(h).strip().lower() for h in header]
    row = [""] * len(header)
    # direct matches
    for k, v in values_by_key.items():
        if v is None:
            continue
        kn = str(k).strip().lower()
        if kn in hnorm:
            row[hnorm.index(kn)] = v
    return row


def write_predictions_to_tracker(ss, picks_rows, tzname, method_name):
    """Append Markov picks into Prediction_Tracker tab without disturbing existing data."""
    # allow both casings
    ws = _open_ws_fuzzy(ss, ["Prediction_Tracker", "prediction_tracker", "Prediction Tracker", "prediction tracker"])
    header = ws.row_values(1)
    if not header:
        raise Exception("Prediction_Tracker has no header row; cannot append safely.")
    run_ts = _now_iso(tzname)

    # common header variants
    def pick_to_text(p):
        """
        Convert an internal pick representation to tracker-friendly text.

        Supports:
          - dict: {'numbers': [...], 'powerball': x} or {'main': [...], 'pb': x}
          - list/tuple: [n1, n2, ..., '-', pb] where '-' may separate main/special
          - strings: already formatted
        """
        if p is None:
            return ""
        if isinstance(p, str):
            return p.strip()

        def _as_int(x):
            if x is None:
                return None
            if isinstance(x, (int,)):
                return int(x)
            # handle floats that are whole numbers
            if isinstance(x, float):
                if x.is_integer():
                    return int(x)
                return None
            sx = str(x).strip()
            if sx in ("", "-", "—", "–", "None", "nan", "NaN"):
                return None
            # allow things like "PB: 12" or "12*"
            m = re.search(r"\d+", sx)
            return int(m.group(0)) if m else None

        # dict form
        if isinstance(p, dict):
        # If caller already formatted a pick string, use it directly
        for k in ("pick", "prediction", "pred", "pred_text"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

            main = p.get("main", None)
            if main is None:
                main = p.get("numbers", None)
            if main is None:
                main = p.get("nums", [])
            pb = p.get("powerball", p.get("pb", p.get("special", None)))

            main_nums = [x for x in (_as_int(x) for x in (main or [])) if x is not None]
            pb_int = _as_int(pb)

            s = " ".join(str(x) for x in main_nums)
            if pb_int is not None:
                s = f"{s} - {pb_int}" if s else str(pb_int)
            return s.strip()

        # list/tuple form
        if isinstance(p, (list, tuple)):
            items = list(p)
            pb_val = None

            if "-" in items:
                i = items.index("-")
                main_items = items[:i]
                tail = items[i+1:]
                # if tail has a single numeric, treat as special ball
                tail_ints = [x for x in (_as_int(x) for x in tail) if x is not None]
                if len(tail_ints) == 1:
                    pb_val = tail_ints[0]
                items = main_items
            else:
                # sometimes special is stored as last element in separate field;
                # keep as main-only unless it is explicitly a 1-item tail in dict form.
                pass

            main_nums = [x for x in (_as_int(x) for x in items) if x is not None]
            s = " ".join(str(x) for x in main_nums)
            if pb_val is not None:
                s = f"{s} - {pb_val}" if s else str(pb_val)
            return s.strip()

        # fallback
        return str(p).strip()

    for p in picks_rows:
        game = p.get("game") or p.get("game_name") or ""
        pred_text = pick_to_text(p)

        values = {
            "timestamp": run_ts,
            "run_timestamp": run_ts,
            "run_ts": run_ts,
            "date": run_ts.split("T")[0],
            "game": game,
            "game_name": game,
            "method": method_name,
            "model": method_name,
            "source": method_name,
            "generator": method_name,
            "prediction": pred_text,
            "predictions": pred_text,
            "pick": pred_text,
            "numbers": pred_text,
            "numset": pred_text,
            "set": pred_text,
            "notes": "markov_mc",
        }
        row = _row_from_header(header, values)

        # if nothing mapped for core columns, fall back to appending a minimal row at end
        if all(x == "" for x in row):
            row = [run_ts, game, method_name, pred_text]
        ws.append_row(row, value_input_option="RAW")
def _df_to_ws(ws: gspread.Worksheet, df: pd.DataFrame, clear: bool = True) -> None:
    if clear:
        ws.clear()
    if df.empty:
        ws.update("A1", [["(no data)"]])
        return
    # Convert to strings for Sheets
    out = [list(df.columns)]
    for _, r in df.iterrows():
        out.append([("" if (pd.isna(v) or v is None) else str(v)) for v in r.tolist()])
    ws.update("A1", out)

def _read_draws_from_tab(ss: gspread.Spreadsheet, spec: GameSpec) -> Tuple[pd.DataFrame, List[List[int]]]:
    ws = ss.worksheet(spec.results_tab)
    df = _worksheet_get_df(ws)
    if df.empty:
        return df, []
    date_col = _find_date_col(df)
    num_cols = _infer_number_cols(df)
    if date_col is None or len(num_cols) < spec.main_count:
        # fail loudly with diagnostics
        raise RuntimeError(
            f"[{spec.name}] Could not infer columns. date_col={date_col}, "
            f"num_cols={num_cols}. Columns={list(df.columns)[:30]}"
        )
    # coerce dates and sort
    df["_date"] = _coerce_date_series(df[date_col])
    df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)

    # take needed columns in order; attempt to detect bonus column (Powerball)
    # We'll just take first main_count numbers from num_cols, then if bonus_count, next one.
    use_cols = num_cols[: spec.main_count + spec.bonus_count]
    dnums: List[List[int]] = []
    for _, row in df[use_cols].iterrows():
        vals = []
        for v in row.tolist():
            try:
                vals.append(int(str(v).strip()))
            except Exception:
                vals.append(None)
        if any(x is None for x in vals[:spec.main_count]):
            continue
        if spec.bonus_count and vals[spec.main_count] is None:
            # if missing bonus, skip
            continue
        vals_int = [int(x) for x in vals]
        dnums.append(_normalize_draw(vals_int, spec))
    return df, dnums

def main() -> int:
    tzname = os.getenv("LOCAL_TZ", "America/Chicago")
    method_name = os.getenv("MARKOV_METHOD_NAME", "markov_mc").strip() or "markov_mc"
    try:
        creds_json = _env("GOOGLE_CREDS_JSON")
        sheet_name = _env("SHEET_NAME")
        targets = _parse_json_maybe(os.getenv("MARKOV_TARGETS_JSON", "{}"))
        temp = _env_float("MARKOV_TEMP", 1.0)
        smooth = _env_float("MARKOV_SMOOTH", 1.0)
        seed = _env_int("MARKOV_RANDOM_SEED", 123)
        backtest_draws = _env_int("MARKOV_BACKTEST_DRAWS", 200)
        lookback_min = _env_int("MARKOV_LOOKBACK_MIN", 200)

        print(f"[MARKOV] start={_now_iso(tzname)} tz={tzname}")
        print(f"[MARKOV] config: temp={temp} smooth={smooth} seed={seed} backtest_draws={backtest_draws} lookback_min={lookback_min}")
        print(f"[MARKOV] targets: {targets}")

        random.seed(seed)
        np.random.seed(seed)
        print("[MARKOV] RNG initialized")

        gc = _gspread_client_from_service_account_json(creds_json)
        ss = _open_sheet(gc, sheet_name)
        print(f"[MARKOV] opened spreadsheet: {ss.title} (id={ss.id})")

        picks_rows = []
        backtest_rows = []

        for game_name, spec in GAME_SPECS.items():
            k = int(targets.get(game_name, targets.get(spec.name, 0)) or 0)
            if k <= 0:
                print(f"[MARKOV] {game_name}: skipped (k=0)")
                continue

            df, draws = _read_draws_from_tab(ss, spec)
            print(f"[MARKOV] {game_name}: draws={len(draws)} tab={spec.results_tab}")

            if len(draws) < 5:
                print(f"[MARKOV] {game_name}: not enough draws to model; skipping")
                continue

            picks = generate_picks(draws, spec, k=k, temp=temp, smooth=smooth, seed=seed)
            print(f"[MARKOV] {game_name}: generated {len(picks)} picks")
            for i, p in enumerate(picks, start=1):
                picks_rows.append({
                    "timestamp": _now_iso(tzname),
                    "game": game_name,
                    "rank": i,
                    "pick": _format_pick(p, spec),
                    "method": method_name,
                    "temp": temp,
                    "smooth": smooth,
                })

            bt = backtest(draws, spec, k=k, temp=temp, smooth=smooth, seed=seed,
                          lookback_min=lookback_min, backtest_draws=backtest_draws)
            if bt.empty:
                print(f"[MARKOV] {game_name}: backtest empty (need more history)")
            else:
                # Summary to log
                any3 = bt["any_3plus_main"].mean()
                any4 = bt["any_4plus_main"].mean()
                best_avg = bt["best_main"].mean()
                avg_avg = bt["avg_main"].mean()
                print(f"[MARKOV][BACKTEST] {game_name}: rows={len(bt)} "
                      f"best_main_mean={best_avg:.3f} avg_main_mean={avg_avg:.3f} "
                      f"P(any>=3)={any3:.3%} P(any>=4)={any4:.3%}")
                # Add game column and append
                bt2 = bt.copy()
                bt2.insert(0, "game", game_name)
                backtest_rows.append(bt2)

        # Always append to Prediction_Tracker (keeps existing prediction history intact)
        write_predictions_to_tracker(ss, picks_rows, tzname, method_name=method_name)

        # Optional: keep Markov-specific tabs for inspection/backtest
        keep_tabs = os.getenv("KEEP_MARKOV_TABS", "0").strip() in ("1", "true", "TRUE", "yes", "YES")
        if keep_tabs:
            ws_picks = _ensure_ws(ss, "Markov_Picks", rows=5000, cols=12)
            ws_bt = _ensure_ws(ss, "Markov_Backtest", rows=10000, cols=20)

            picks_df = pd.DataFrame(picks_rows)
            bt_df = pd.concat(backtest_rows, ignore_index=True) if backtest_rows else pd.DataFrame()

            _df_to_ws(ws_picks, picks_df, clear=True)
            _df_to_ws(ws_bt, bt_df, clear=True)

            print(f"[MARKOV] wrote tabs: Markov_Picks rows={len(picks_df)}; Markov_Backtest rows={len(bt_df)}")
        else:
            print(f"[MARKOV] appended {len(picks_rows)} rows to Prediction_Tracker")

        print(f"[MARKOV] done={_now_iso(tzname)}")
        return 0

    except Exception as e:
        print("[MARKOV][ERROR]", str(e))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())