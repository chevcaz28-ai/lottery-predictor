
import math, random, hashlib, os
from typing import List, Any

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
    return [0.0] + [e for e in softmax(scores[1:], temperature)]

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
    return sorted(chosen)

def markov_mc_method(rows_hist: List[List[Any]], need: int, lo: int, hi: int, temp: float=0.9, smooth: float=1.0, seed_mode: str="auto") -> List[int]:
    history: List[List[int]] = []
    for r in rows_hist:
        draw = []
        for v in r[1:1+need]:
            try:
                iv = int(v)
                if lo <= iv <= hi:
                    draw.append(iv)
            except Exception:
                continue
        if len(draw) >= need:
            history.append(sorted(draw))
    if not history:
        return sorted(random.sample(range(lo, hi+1), need))

    latest_date = str(rows_hist[0][0])
    rng = random.Random()

    sm = str(seed_mode).strip().lower() if seed_mode is not None else "auto"
    if sm in ("", "none", "null"):
        sm = "auto"

    if sm == "auto":
        # Stable per-draw randomness: hash the latest draw date
        h = hashlib.sha1(latest_date.encode("utf-8")).hexdigest()
        rng.seed(int(h[:12], 16))
    elif sm == "fixed":
        # Fixed seed across runs (use env MARKOV_RANDOM_SEED if set, else fallback)
        base = os.getenv("MARKOV_RANDOM_SEED", "").strip() or "123"
        try:
            rng.seed(int(base))
        except Exception:
            pass
    elif sm == "draw":
        # Derived seed per draw: base seed + hash(latest draw date)
        base = os.getenv("MARKOV_RANDOM_SEED", "").strip() or "123"
        try:
            base_seed = int(base)
        except Exception:
            base_seed = 123
        h = hashlib.sha1(latest_date.encode("utf-8")).hexdigest()
        rng.seed((base_seed + int(h[:12], 16)) % (2**31 - 1))
    else:
        # If someone passes a numeric seed string, honor it
        try:
            rng.seed(int(seed_mode))
        except Exception:
            pass

    T, prior = build_transition_and_prior(history, hi, smooth)
    last_set = history[0]
    probs = next_probs_from_last(last_set, T, prior, temp)
    cands = list(range(1, hi+1))
    weights = probs[1:]
    pick = pick_weighted_without_replacement(cands, weights, need, rng)
    return pick

def freq_method(rows: List[List[Any]], need:int, lo:int, hi:int, window:int=50) -> List[int]:
    hist = {}
    for r in rows[:window]:
        for v in r[1:1+need]:
            try:
                v = int(v)
                if lo<=v<=hi:
                    hist[v] = hist.get(v, 0)+1
            except: 
                pass
    ranked = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [k for k,_ in ranked][:need]
    s = set(base)
    import random as _rnd
    while len(s) < need:
        x = _rnd.randint(lo, hi)
        if x not in s: s.add(x)
    return sorted(s)

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
    s = set(base)
    import random as _rnd
    while len(s) < need:
        x = _rnd.randint(lo, hi)
        if x not in s: s.add(x)
    return sorted(s)
