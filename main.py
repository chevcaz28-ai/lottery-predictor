#!/usr/bin/env python3

"""
main.py (updated)

What this version adds:
- Robust env flag parsing (ENABLE_LLM_METHOD, ENABLE_BASELINE, etc.)
- Clear diagnostics showing whether LLM is actually enabled/usable
- Recent-history de-dup so you don't see the same numbers every cycle
- Optional non-deterministic RNG via MARKOV_RANDOM_SEED=auto|<int>
- Safe Google Sheets I/O with minimal retries

This is designed to be a drop-in replacement, but if your repository
has custom prediction generation, you can plug it into the
`generate_predictions_for_game()` function below.
"""
import os
import sys
import json
import time
import random
import math
import re
from datetime import datetime, timezone, date
from typing import List, Tuple, Dict, Optional, Set, Any

import numpy as np

# ---- Optional deps; gspread required if you use Google Sheets storage ----
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception as _e:
    gspread = None
    Credentials = None

# -----------------------------------------------------------------------------
# Additional helpers and legacy functions from v4.5.x
# These functions implement more sophisticated prediction methods (frequency,
# recency, markov) and adaptive weighting similar to the original script.  They
# coexist with the simplified logic in this file so that you can choose
# between random picks and weighted method‑based picks without breaking
# existing behaviour.

# Standard tracker column layout (8 columns: A..H)
TRACKER_COLS = [
    "Timestamp",
    "Game",
    "Next Draw Date",
    "Prediction",
    "Method",
    "Win Count",
    "Matches",
    "Match Count",
]

def normalize_date(s: Optional[str]) -> str:
    """Normalize date strings to YYYY‑MM‑DD.  Accepts MM/DD/YYYY or YYYY‑MM‑DD."""
    if s is None:
        return ""
    ss = str(s).strip()
    if not ss:
        return ""
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", ss)
    if m:
        mm, dd, yy = m.groups()
        try:
            return f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}"
        except Exception:
            return ss
    m2 = re.match(r"^(\d{4})-(\d{2})-(\d{2})", ss)
    if m2:
        return ss[:10]
    return ss

def today_local_str() -> str:
    """Return today's date in YYYY‑MM‑DD format in local time zone."""
    return datetime.now().astimezone().strftime("%Y-%m-%d")

def timestamp_local_str() -> str:
    """Return current timestamp string with timezone info."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

def unique_combo(nums: List[int], need: int, lo: int, hi: int) -> List[int]:
    """Return a sorted list of unique numbers of length `need` within [lo, hi]."""
    s = sorted(set(n for n in nums if isinstance(n, int) and lo <= n <= hi))
    while len(s) < need:
        x = random.randint(lo, hi)
        if x not in s:
            s.append(x)
    return sorted(s)[:need]

def mutate(nums: List[int], lo: int, hi: int) -> List[int]:
    """Randomly replace one element in the list with a new unique value."""
    s = list(sorted(set(int(x) for x in nums if str(x).isdigit())))
    if not s:
        return s
    for _ in range(50):
        idx = random.randrange(0, len(s))
        cand = random.randint(lo, hi)
        if cand not in s:
            s[idx] = cand
            break
    return sorted(s)

def freq_method(rows: List[List[Any]], need: int, lo: int, hi: int, window: int = 50) -> List[int]:
    """Frequency‑based method: most common numbers in recent draws."""
    hist: Dict[int, int] = {}
    for r in rows[:window]:
        for v in r[1:1 + need]:
            try:
                iv = int(v)
                if lo <= iv <= hi:
                    hist[iv] = hist.get(iv, 0) + 1
            except Exception:
                pass
    ranked = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [k for k, _ in ranked][:need]
    return unique_combo(base, need, lo, hi)

def recency_method(rows: List[List[Any]], need: int, lo: int, hi: int, decay: float = 0.9) -> List[int]:
    """Recency‑weighted method: favour numbers drawn more recently."""
    hist: Dict[int, float] = {}
    w = 1.0
    for r in rows:
        for v in r[1:1 + need]:
            try:
                iv = int(v)
                if lo <= iv <= hi:
                    hist[iv] = hist.get(iv, 0) + w
            except Exception:
                pass
        w *= decay
    ranked = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [k for k, _ in ranked][:need]
    return unique_combo(base, need, lo, hi)

def markov1_method(rows: List[List[Any]], need: int, lo: int, hi: int) -> List[int]:
    """Simple first‑order Markov method based on transitions between numbers."""
    seq: List[int] = []
    for r in rows:
        for v in r[1:1 + need]:
            try:
                iv = int(v)
            except Exception:
                continue
            if lo <= iv <= hi:
                seq.append(iv)
    # Build transition counts
    trans: Dict[int, Dict[int, int]] = {i: {} for i in range(lo, hi + 1)}
    for a, b in zip(seq, seq[1:]):
        trans[a][b] = trans[a].get(b, 0) + 1
    # Seeds: numbers from the most recent draws
    seeds: List[int] = []
    for r in rows[:3]:
        for v in r[1:1 + need]:
            try:
                iv = int(v)
            except Exception:
                continue
            if lo <= iv <= hi and iv not in seeds:
                seeds.append(iv)
    # Aggregate candidate counts
    candidate_counts: Dict[int, int] = {}
    for s in seeds:
        for nxt, cnt in trans.get(s, {}).items():
            candidate_counts[nxt] = candidate_counts.get(nxt, 0) + cnt
    ranked = sorted(candidate_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [k for k, _ in ranked[:need]]
    base_set = set(base)
    while len(base_set) < need:
        x = random.randint(lo, hi)
        if x not in base_set:
            base_set.add(x)
    return sorted(base_set)

def pick_powerball(rows: List[List[Any]]) -> int:
    """Pick the most common Powerball from recent draws (fallback random)."""
    hist: Dict[int, float] = {}
    w = 1.0
    for r in rows:
        try:
            pb = int(r[6])
            hist[pb] = hist.get(pb, 0) + w
        except Exception:
            pass
        w *= 0.95
    if not hist:
        return random.randint(1, 26)
    return sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def fmt_prediction_str(game: str, nums: List[int], special: Optional[int] = None) -> str:
    """Format a prediction string consistent with the simple generator."""
    main = " ".join(str(n) for n in sorted(nums))
    if game == "Powerball" and special is not None:
        return f"{main} | PB {special}"
    return main

def get_or_create_worksheet(ss: Optional[gspread.Spreadsheet], title: str, rows: int = 2000, cols: int = 8):
    """Return existing worksheet or create a new one if missing."""
    if ss is None:
        return None
    try:
        ws = ss.worksheet(title)
    except Exception:
        ws = ss.add_worksheet(title, rows=rows, cols=cols)
    return ws

def normalize_tracker(ss: Optional[gspread.Spreadsheet]):
    """Ensure Prediction_Tracker has the correct header and column count."""
    if ss is None:
        return
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    if ws is None:
        return
    try:
        # Write header
        ws.update(values=[TRACKER_COLS], range_name="A1")
    except Exception:
        pass
    try:
        vals = ws.get_all_values()
    except Exception:
        return
    body = vals[1:] if len(vals) >= 2 else []
    if not body:
        return
    need_cols = len(TRACKER_COLS)
    changed = False
    norm_rows = []
    for r in body:
        if len(r) != need_cols:
            changed = True
        norm_rows.append((list(r) + [""] * (need_cols - len(r)))[:need_cols])
    if changed:
        # Compute end column letter
        def col_letter(n: int) -> str:
            s = ""
            while n > 0:
                n, r = divmod(n - 1, 26)
                s = chr(65 + r) + s
            return s or "A"
        end_col = col_letter(need_cols)
        try:
            ws.update(values=norm_rows, range_name=f"A2:{end_col}{len(norm_rows) + 1}")
        except Exception:
            pass

def adaptive_weights(ss: Optional[gspread.Spreadsheet], game: str) -> Dict[str, float]:
    """Compute adaptive weights for each method based on tracker history."""
    if ss is None:
        return {}
    ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
    if ws is None:
        return {}
    # Normalize tracker before reading
    normalize_tracker(ss)
    try:
        recs = ws.get_all_records()
    except Exception:
        return {}
    stats: Dict[str, Dict[str, float]] = {}
    allowed_methods: Set[str] = {
        "last_draw_baseline",
        "freq50",
        "recency",
        "markov1",
        "llm_gpt",
        "filler_random",
    }
    for r in recs:
        if str(r.get("Game", "")).strip() != game:
            continue
        md = str(r.get("Method", "") if r.get("Method", "") is not None else "").strip()
        if md not in allowed_methods:
            continue
        try:
            mc = int(str(r.get("Match Count", "0")).strip() or "0")
        except Exception:
            mc = 0
        try:
            wc = int(str(r.get("Win Count", "0")).strip() or "0")
        except Exception:
            wc = 0
        if md not in stats:
            stats[md] = {"cnt": 0, "match_sum": 0.0, "wins": 0.0}
        stats[md]["cnt"] += 1
        stats[md]["match_sum"] += mc
        stats[md]["wins"] += wc
    weights: Dict[str, float] = {}
    for md, s in stats.items():
        if s["cnt"] == 0:
            continue
        score = (s["match_sum"] / s["cnt"]) + 5.0 * s["wins"]
        weights[md] = score
    if not weights:
        return {}
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}

def disabled_methods_for_game(game: str) -> set:
    """Return a set of method names disabled for this game via environment variables."""
    env_map = {
        "Powerball": "DISABLE_METHODS_POWERBALL",
        "Megabucks": "DISABLE_METHODS_MEGABUCKS",
        "SuperCash": "DISABLE_METHODS_SUPERCASH",
        "Badger 5": "DISABLE_METHODS_BADGER5",
    }
    env = env_map.get(game, "")
    raw = os.getenv(env, "") or ""
    return {m.strip() for m in raw.split(",") if m.strip()}

def safe_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss.isdigit():
        return int(ss)
    m = re.match(r"^\s*(\d+)", ss)
    if m:
        return int(m.group(1))
    return None

def target_total_for_game(game: str, enabled_method_count: int) -> int:
    mapping = {
        "Powerball": PREDICTIONS_POWERBALL,
        "Megabucks": PREDICTIONS_MEGABUCKS,
        "SuperCash": PREDICTIONS_SUPERCASH,
        "Badger 5": PREDICTIONS_BADGER5,
    }
    specific_raw = mapping.get(game)
    specific = safe_int(specific_raw)
    global_default = safe_int(PREDICTIONS_PER_GAME)
    if specific is not None:
        return max(specific, enabled_method_count)
    if global_default is not None:
        return max(global_default, enabled_method_count)
    return enabled_method_count

def predictions_dry_run() -> bool:
    """Check if PREDICTIONS_DRY_RUN is truthy, in which case predictions are not appended."""
    return os.getenv("PREDICTIONS_DRY_RUN", "0").strip().lower() in ("1", "true", "yes")

def llm_pick_numbers(game: str, rows_hist: List[List[Any]]) -> Tuple[List[int], Optional[int]]:
    """Use the OpenAI API to generate picks.  Returns (mains, special)."""
    if not LLM_OK:
        return [], None
    if _openai_client is None:
        return [], None
    # Prepare historical context (up to 100 rows)
    hist_lines: List[str] = []
    for r in rows_hist[:100]:
        date_str = str(r[0])
        if game == "Powerball":
            nums = [r[1], r[2], r[3], r[4], r[5]]
            pb = r[6] if len(r) > 6 else None
            hist_lines.append(f"{date_str}: {nums} PB:{pb}")
        elif game == "Megabucks":
            nums = [r[i] for i in range(1, 1 + GAME_RULES[game]["main"][2])]
            hist_lines.append(f"{date_str}: {nums}")
        elif game == "SuperCash":
            nums = [r[i] for i in range(1, 1 + GAME_RULES[game]["main"][2])]
            hist_lines.append(f"{date_str}: {nums}")
        elif game == "Badger 5":
            nums = [r[i] for i in range(1, 1 + GAME_RULES[game]["main"][2])]
            hist_lines.append(f"{date_str}: {nums}")
    hist_text = "\n".join(hist_lines)
    # Construct prompts
    system = (
        "You generate lottery number-set predictions strictly in the required ranges. "
        "Return only JSON like {\"mains\":[...],\"special\":<int or null>} with no extra text."
    )
    if game == "Powerball":
        rule = "Pick 5 distinct mains between 1 and 69, and 1 Powerball between 1 and 26."
    elif game == "Megabucks":
        lo, hi, need = GAME_RULES[game]["main"]
        rule = f"Pick {need} distinct mains between {lo} and {hi}. No special ball."
    elif game == "SuperCash":
        lo, hi, need = GAME_RULES[game]["main"]
        rule = f"Pick {need} distinct mains between {lo} and {hi}. No special ball."
    elif game == "Badger 5":
        lo, hi, need = GAME_RULES[game]["main"]
        rule = f"Pick {need} distinct mains between {lo} and {hi}. No special ball."
    else:
        return [], None
    user = (
        f"Game: {game}\n"
        f"Rules: {rule}\n"
        f"Recent results (newest first):\n{hist_text}\n"
        'Respond ONLY with JSON: {"mains":[int,...],"special":null or int}'
    )
    mains: List[int] = []
    special: Optional[int] = None
    try:
        resp = _openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=LLM_TEMP,
        )
        content = resp.choices[0].message.content.strip()
        js = json.loads(content)
        mains = [int(x) for x in js.get("mains", []) if isinstance(x, (int, str)) and str(x).isdigit()]
        sp = js.get("special", None)
        special = int(sp) if sp is not None and str(sp).isdigit() else None
    except Exception:
        return [], None
    # Clip and ensure uniqueness
    def clip_and_unique(nums: List[int], need: int, lo: int, hi: int) -> List[int]:
        s: List[int] = []
        for x in nums:
            try:
                xi = int(x)
                if lo <= xi <= hi and xi not in s:
                    s.append(xi)
            except Exception:
                pass
        while len(s) < need:
            xi = random.randint(lo, hi)
            if xi not in s:
                s.append(xi)
        return sorted(s)[:need]
    if game == "Powerball":
        mains = clip_and_unique(mains, 5, 1, 69)
        if special is None or not (1 <= special <= 26):
            special = random.randint(1, 26)
    else:
        lo, hi, need = GAME_RULES[game]["main"]
        mains = clip_and_unique(mains, need, lo, hi)
        special = None
    return mains, special

def load_results_history(ss: Optional[gspread.Spreadsheet], game: str, max_rows: int = 200) -> List[List[Any]]:
    """Load historical result rows for a given game from its Results worksheet."""
    if ss is None:
        return []
    sheet_map = {
        "Powerball": "Powerball_Results",
        "Megabucks": "Megabucks_Results",
        "SuperCash": "SuperCash_Results",
        "Badger 5": "Badger5_Results",
    }
    ws_name = sheet_map.get(game)
    if not ws_name:
        return []
    try:
        ws = ss.worksheet(ws_name)
    except Exception:
        return []
    try:
        values = ws.get_all_values()
    except Exception:
        return []
    # Skip header
    body = values[1:] if values and len(values) >= 1 else []
    # Normalize date strings and sort newest to oldest
    def date_key(r):
        return normalize_date(r[0])
    rows_sorted = sorted(body, key=lambda r: normalize_date(r[0]), reverse=True)
    return rows_sorted[:max_rows]

def generate_predictions_for_game(game: str, ss: Optional[gspread.Spreadsheet], target: int, recent: Optional[Set[str]] = None) -> List[str]:
    """Generate predictions using multiple methods and adaptive weighting.

    Parameters
    ----------
    game : str
        The game name (e.g., "Powerball").
    ss : gspread.Spreadsheet or None
        The spreadsheet instance for reading history and tracker.  If None, falls back to random picks.
    target : int
        Desired number of predictions for this game.
    recent : set[str], optional
        A set of recent prediction strings from the <Game>_Predictions sheet to avoid repeats.
    Returns
    -------
    List[str]
        A list of formatted prediction strings.
    """
    # Determine game parameters
    if game not in GAME_RULES:
        return []
    lo, hi, need = GAME_RULES[game]["main"]
    rows_hist = load_results_history(ss, game, max_rows=200)
    # Determine baseline numbers (last draw)
    baseline_nums: List[int] = []
    if rows_hist:
        try:
            baseline_nums = [int(x) for x in rows_hist[0][1:1 + need] if str(x).isdigit()]
        except Exception:
            baseline_nums = []
    # Determine special for Powerball from history
    special_pb = pick_powerball(rows_hist) if game == "Powerball" else None
    # Build methods dictionary: method name -> (mains, special)
    methods: Dict[str, Tuple[List[int], Optional[int]]] = {}
    # Apply per‑game method disables
    _dis = disabled_methods_for_game(game)
    # Baseline method
    if ENABLE_BASELINE and baseline_nums and "last_draw_baseline" not in _dis:
        methods["last_draw_baseline"] = (baseline_nums, None)
    # Frequency method
    if "freq50" not in _dis:
        methods["freq50"] = (freq_method(rows_hist, need, lo, hi, window=50), None)
    # Recency method
    if "recency" not in _dis:
        methods["recency"] = (recency_method(rows_hist, need, lo, hi, decay=0.92), None)
    # Markov method
    if "markov1" not in _dis:
        methods["markov1"] = (markov1_method(rows_hist, need, lo, hi), None)
    # LLM method
    if ENABLE_LLM and LLM_OK and "llm_gpt" not in _dis:
        mains_llm, special_llm = llm_pick_numbers(game, rows_hist)
        if mains_llm:
            methods["llm_gpt"] = (mains_llm, special_llm if game == "Powerball" else None)
    enabled_methods = list(methods.keys())
    # Compute target total (at least number of enabled methods)
    total_target = target_total_for_game(game, len(enabled_methods))
    # Build set of existing predictions for this draw (today) to avoid duplicates
    existing_preds_for_draw: Set[str] = set()
    if ss is not None:
        try:
            ws = get_or_create_worksheet(ss, "Prediction_Tracker", rows=20000, cols=len(TRACKER_COLS))
            normalize_tracker(ss)
            recs = ws.get_all_records()
            next_key = today_local_str()
            for r in recs:
                if str(r.get("Game", "")).strip() == game and normalize_date(r.get("Next Draw Date", "")) == next_key:
                    pred = str(r.get("Prediction", "")).strip()
                    if pred:
                        existing_preds_for_draw.add(pred)
        except Exception:
            pass
    remaining = max(0, total_target - len(existing_preds_for_draw))
    picks: List[str] = []
    written_preds: Set[str] = set()
    timestamp = timestamp_local_str()
    # Helper to emit a pick
    def emit(md: str, nums: List[int], sp: Optional[int]) -> bool:
        # Compose prediction string
        pred_str = fmt_prediction_str(game, nums, sp if (md == "llm_gpt" and sp is not None) else (special_pb if game == "Powerball" else None))
        # Avoid duplicates from tracker, within this run, and from recent predictions in the predictions sheet
        if pred_str in existing_preds_for_draw or pred_str in written_preds or (recent and pred_str in recent):
            return False
        picks.append(pred_str)
        written_preds.add(pred_str)
        return True
    # Emit one pick per enabled method
    for md in enabled_methods:
        if remaining <= 0:
            break
        nums, sp = methods[md]
        if emit(md, nums, sp):
            remaining -= 1
    # Weighted extra variants
    if remaining > 0 and enabled_methods:
        weights = adaptive_weights(ss, game)
        # Normalize weights among enabled methods
        if weights:
            total_w = sum(weights.get(m, 0.0) for m in enabled_methods)
            if total_w > 0:
                w_map = {m: weights.get(m, 0.0) / total_w for m in enabled_methods}
            else:
                w_map = {m: 1.0 / len(enabled_methods) for m in enabled_methods}
        else:
            w_map = {m: 1.0 / len(enabled_methods) for m in enabled_methods}
        def weighted_pick(wdict: Dict[str, float]) -> str:
            r = random.random()
            cum = 0.0
            for k, v in wdict.items():
                cum += v
                if r <= cum:
                    return k
            return enabled_methods[0]
        base_cache = {md: list(methods[md][0]) for md in enabled_methods}
        base_sp = {md: (methods[md][1] if md == "llm_gpt" else None) for md in enabled_methods}
        tries = 0
        max_tries = 800
        while remaining > 0 and tries < max_tries:
            tries += 1
            md = weighted_pick(w_map)
            base_nums = base_cache.get(md, [])
            if not base_nums:
                continue
            variant = mutate(base_nums, lo, hi)
            sp = base_sp.get(md, None)
            if emit(md, variant, sp):
                remaining -= 1
        # Fallback round robin
        if remaining > 0:
            non_empty = [m for m, n in base_cache.items() if n]
            rr = 0
            safety = 0
            while remaining > 0 and non_empty and safety < 1000:
                safety += 1
                md = non_empty[rr % len(non_empty)]
                rr += 1
                variant = mutate(base_cache[md], lo, hi)
                sp = base_sp.get(md, None)
                if emit(md, variant, sp):
                    remaining -= 1
        # Final filler random combos
        if remaining > 0:
            def fresh_combo(need: int, lo: int, hi: int) -> List[int]:
                s: Set[int] = set()
                while len(s) < need:
                    s.add(random.randint(lo, hi))
                return sorted(s)
            safety2 = 0
            while remaining > 0 and safety2 < 3000:
                safety2 += 1
                nums = fresh_combo(need, lo, hi)
                if emit("filler_random", nums, None):
                    remaining -= 1
    return picks

# -------------------- ENV / FLAGS --------------------

def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"): return True
    if v in ("0", "false", "no", "n", "off", ""): return False
    return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except Exception:
        return default

def choose_seed() -> Optional[int]:
    raw = os.getenv("MARKOV_RANDOM_SEED", "auto").strip().lower()
    if raw in ("", "auto", "random", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None

ENABLE_LLM = env_flag("ENABLE_LLM_METHOD", False)
ENABLE_BASELINE = env_flag("ENABLE_BASELINE", False)  # default FALSE to avoid drowning LLM/Markov
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_TEMP = float(os.getenv("LLM_TEMP", "0.9"))
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/Chicago")

PREDICTIONS_PER_GAME = env_int("PREDICTIONS_PER_GAME", 10)
PREDICTIONS_POWERBALL  = env_int("PREDICTIONS_POWERBALL", 5)
PREDICTIONS_MEGABUCKS  = env_int("PREDICTIONS_MEGABUCKS", 10)
PREDICTIONS_SUPERCASH  = env_int("PREDICTIONS_SUPERCASH", 10)
PREDICTIONS_BADGER5    = env_int("PREDICTIONS_BADGER5", 5)

# RNG seeding for diversity/noise
_seed = choose_seed()
if _seed is None:
    np.random.seed(None)
    random.seed()
else:
    np.random.seed(_seed)
    random.seed(_seed)

# LLM gate
LLM_OK = False
LLM_REASON = "disabled"
_openai_client = None

if ENABLE_LLM:
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            LLM_REASON = "no OPENAI_API_KEY"
        else:
            _openai_client = OpenAI(api_key=key)
            LLM_OK = True
            LLM_REASON = "ready"
    except Exception as e:
        LLM_OK = False
        LLM_REASON = f"import/error: {type(e).__name__}: {e}"

print(f"[DIAG] LLM_CFG enabled={ENABLE_LLM} provider='{LLM_PROVIDER}' model='{LLM_MODEL}' temp={LLM_TEMP}")
print(f"[DIAG] LLM_OK={LLM_OK} reason={LLM_REASON}")
print(f"[DIAG] baseline enabled={ENABLE_BASELINE}")
print(f"[DIAG] RNG seed={_seed} (MARKOV_RANDOM_SEED='{os.getenv('MARKOV_RANDOM_SEED', 'auto')}')")


# -------------------- GAME RANGES (WI games) --------------------
# Adjust if your jurisdiction differs.
GAME_RULES = {
    "Powerball": {"main": (1, 69, 5), "power": (1, 26, 1)},           # 5 of 1..69 + 1 PB 1..26
    "Megabucks": {"main": (1, 49, 6)},                                 # 6 of 1..49
    "SuperCash": {"main": (1, 39, 4)},                                 # 4 of 1..39
    "Badger 5": {"main": (1, 31, 5)},                                  # 5 of 1..31
}

# -------------------- SHEETS HELPERS --------------------
def _connect_sheet() -> Optional[gspread.Spreadsheet]:
    sheet_name = os.getenv("SHEET_NAME", "").strip()
    if not sheet_name:
        print("::warning::SHEET_NAME not set; running without Sheets I/O.")
        return None
    if gspread is None:
        print("::warning::gspread/google-auth not installed; running without Sheets I/O.")
        return None

    raw_creds = os.getenv("GOOGLE_CREDS_JSON", "").strip()
    if not raw_creds:
        print("::warning::GOOGLE_CREDS_JSON missing; running without Sheets I/O.")
        return None

    try:
        sa_info = json.loads(raw_creds)
        scopes = [
           "https://www.googleapis.com/auth/spreadsheets",
           "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)
        ss = gc.open(sheet_name)
        print(f"[DIAG] Opened spreadsheet '{sheet_name}'")
        return ss
    except Exception as e:
        print(f"::warning::Failed to open Google Sheet: {e}")
        return None

def _touch_debug(ws_title: str, ss: gspread.Spreadsheet) -> None:
    try:
        try:
            ws = ss.worksheet(ws_title)
        except Exception:
            ws = ss.add_worksheet(ws_title, rows=100, cols=5)
        now = datetime.now(timezone.utc).isoformat()
        ws.append_row(["touch", now])
        print(f"[DIAG] Wrote a {ws_title} row successfully.")
    except Exception as e:
        print(f"::warning::Failed to write {ws_title}: {e}")

# -------------------- RECENT HISTORY DEDUPE --------------------
def fmt_pick(game: str, nums: List[int], extra: Optional[int] = None) -> str:
    if game == "Powerball":
        main = " ".join(map(str, sorted(nums)))
        return f"{main} | PB {extra}"
    else:
        return " ".join(map(str, sorted(nums)))

def load_recent_set(ss: Optional[gspread.Spreadsheet], game: str, max_rows: int = 200) -> Set[str]:
    if ss is None:
        return set()
    title = f"{game}_Predictions"
    try:
        ws = ss.worksheet(title)
    except Exception:
        return set()

    try:
        values = ws.get_all_values()[-max_rows:]
        s: Set[str] = set()
        for row in values:
            # expect first col contains the pick string
            if row and row[0].strip():
                s.add(row[0].strip())
        print(f"[DIAG] recent[{game}] unique={len(s)} (from last {len(values)} rows)")
        return s
    except Exception as e:
        print(f"::warning::Failed to read recent predictions for {game}: {e}")
        return set()

def write_predictions(ss: Optional[gspread.Spreadsheet], game: str, picks: List[str]) -> None:
    if ss is None:
        for p in picks:
            print(f"[{game}] {p}")

# -------------------- PREDICTION TRACKER (optional) --------------------
def append_to_prediction_tracker(ss: Optional[gspread.Spreadsheet], game: str, picks: List[str]) -> None:
    """Append generated picks to the Prediction_Tracker worksheet.

    This helper mirrors a subset of the original v4.5.x behavior: it logs each
    prediction with a timestamp, game name, and a placeholder method in a
    dedicated `Prediction_Tracker` tab.  If the sheet or header row does not
    exist, it will be created.  The tracker uses eight columns:
    Timestamp | Game | Next Draw Date | Prediction | Method | Win Count | Matches | Match Count.

    Parameters
    ----------
    ss: gspread.Spreadsheet or None
        The opened Google Sheets spreadsheet.  If None, nothing is logged.
    game: str
        The name of the game (e.g., "Powerball").
    picks: List[str]
        A list of pick strings generated for the given game.
    """
    # If Sheets I/O isn't available or there are no picks, simply return
    if ss is None or not picks:
        return
    try:
        # Locate or create the Prediction_Tracker worksheet.  We use 8 columns
        # (A..H) which aligns with the historic tracker layout.
        try:
            ws = ss.worksheet("Prediction_Tracker")
        except Exception:
            ws = ss.add_worksheet("Prediction_Tracker", rows=2000, cols=8)

        # Define the standard header for the tracker
        header = [
            "Timestamp",
            "Game",
            "Next Draw Date",
            "Prediction",
            "Method",
            "Win Count",
            "Matches",
            "Match Count",
        ]
        # Ensure the header row exists and matches our specification
        try:
            values = ws.get_all_values()
            if not values:
                ws.update(values=[header], range_name="A1")
            else:
                # If the header differs in length or content, update row 1
                if len(values[0]) < len(header) or any(values[0][i] != header[i] for i in range(min(len(values[0]), len(header)))):
                    ws.update(values=[header], range_name="A1")
        except Exception:
            # We ignore read errors; we'll attempt to write the header anyway
            try:
                ws.update(values=[header], range_name="A1")
            except Exception:
                pass

        # Prepare rows for each pick.  We use the local timezone for the timestamp.
        # The next draw date is left blank; method is a placeholder to indicate
        # this origin.  Win Count, Matches and Match Count are left blank and
        # will be filled when results are processed.
        timestamp = datetime.now().astimezone().isoformat()
        method_name = "random_main"
        new_rows = [
            [timestamp, game, "", pick, method_name, "0", "", ""]
            for pick in picks
        ]
        # Append the new rows to the tracker
        ws.append_rows(new_rows)
    except Exception as e:
        # If any exception occurs, log a warning to console but continue
        print(f"::warning::Failed to append to Prediction_Tracker: {e}")
        return
    title = f"{game}_Predictions"
    try:
        try:
            ws = ss.worksheet(title)
        except Exception:
            ws = ss.add_worksheet(title, rows=2000, cols=3)
        rows = [[p, datetime.now(timezone.utc).isoformat()] for p in picks]
        ws.append_rows(rows)
        print(f"[{game}] appended {len(rows)} predictions.")
    except Exception as e:
        print(f"::warning::Failed to write predictions for {game}: {e}")
        # still log to console
        for p in picks:
            print(f"[{game}] {p}")

# -------------------- GENERATION --------------------
def gen_random_unique(low: int, high: int, k: int) -> List[int]:
    return random.sample(range(low, high + 1), k)

def generate_one_pick(game: str) -> str:
    rules = GAME_RULES[game]
    main_low, main_high, main_k = rules["main"]
    main_nums = gen_random_unique(main_low, main_high, main_k)
    extra = None
    if game == "Powerball":
        pb_low, pb_high, pb_k = rules["power"]
        extra = gen_random_unique(pb_low, pb_high, pb_k)[0]
    return fmt_pick(game, main_nums, extra)

def generate_predictions_simple(game: str, target: int, recent: Set[str]) -> List[str]:
    """Simple diversified generator with recent-history de-dup.

    This fallback generates random unique picks for a game while avoiding
    duplicates with recent picks and within itself.  It is retained for
    completeness but is not used when advanced methods are available.
    """
    out: List[str] = []
    guard = 0
    while len(out) < target and guard < 10000:
        guard += 1
        candidate = generate_one_pick(game)
        if candidate in recent or candidate in out:
            continue
        out.append(candidate)
    return out

# -------------------- MAIN --------------------
def main():
    ss = _connect_sheet()
    if ss is not None:
        _touch_debug("Debug_Touch", ss)

    # Determine targets (env overrides)
    targets = {
        "Powerball": PREDICTIONS_POWERBALL or PREDICTIONS_PER_GAME,
        "Megabucks": PREDICTIONS_MEGABUCKS or PREDICTIONS_PER_GAME,
        "SuperCash": PREDICTIONS_SUPERCASH or PREDICTIONS_PER_GAME,
        "Badger 5": PREDICTIONS_BADGER5 or PREDICTIONS_PER_GAME,
    }

    # Shuffle game order for extra variety
    games = list(targets.keys())
    random.shuffle(games)

    total_emitted = 0
    for game in games:
        target = int(targets[game])
        if target <= 0: 
            continue
        recent = load_recent_set(ss, game, max_rows=200)
        # Generate predictions using advanced methods.  Fallback to simple
        # generator if the spreadsheet is not available (ss is None).
        picks = []
        if target > 0:
            if ss is not None:
                picks = generate_predictions_for_game(game, ss, target, recent)
            else:
                picks = generate_predictions_simple(game, target, recent)
        # Write to the per-game predictions worksheet
        write_predictions(ss, game, picks)
        # Record these picks into the Prediction_Tracker.  This mirrors legacy
        # behaviour and allows later evaluation of win counts.  If the tracker
        # sheet is unavailable, this call silently does nothing.
        append_to_prediction_tracker(ss, game, picks)
        total_emitted += len(picks)

    print(f"Success! main.py finished. Emitted total={total_emitted} predictions.")

if __name__ == "__main__":
    main()
