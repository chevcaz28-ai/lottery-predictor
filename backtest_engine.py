from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


WHITE_COLS = ["W1", "W2", "W3", "W4", "W5"]
PB_COL = "PB"
DATE_COL = "Date"


@dataclass(frozen=True)
class BacktestMetrics:
    draws_tested: int
    avg_white_matches: float
    pb_hit_rate: float
    avg_total_score: float


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in [DATE_COL, *WHITE_COLS, PB_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_by_date(df: pd.DataFrame, split_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _validate_columns(df)
    d = df.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL])
    d = d.sort_values(DATE_COL).reset_index(drop=True)
    split_idx = int(len(d) * split_ratio)
    return d.iloc[:split_idx].reset_index(drop=True), d.iloc[split_idx:].reset_index(drop=True)


def build_frequency_recency_weights(
    train_df: pd.DataFrame,
    recent_n: int = 100,
    alpha: float = 0.7,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build white-ball and PB weights using a blend of long-run frequency and recency.

    alpha=0.7 means 70% long-run + 30% recent_n window.
    """
    _validate_columns(train_df)

    long_white = pd.Series(train_df[WHITE_COLS].to_numpy().flatten()).value_counts()
    recent_white = pd.Series(train_df.tail(recent_n)[WHITE_COLS].to_numpy().flatten()).value_counts()
    white = (long_white * alpha).add(recent_white * (1 - alpha), fill_value=0).sort_index()
    white = white[white > 0]

    long_pb = train_df[PB_COL].value_counts()
    recent_pb = train_df.tail(recent_n)[PB_COL].value_counts()
    pb = (long_pb * alpha).add(recent_pb * (1 - alpha), fill_value=0).sort_index()
    pb = pb[pb > 0]

    return white, pb


def generate_ticket(
    white_weights: pd.Series,
    pb_weights: pd.Series,
    rng: np.random.Generator,
) -> Tuple[Tuple[int, ...], int]:
    whites = rng.choice(
        white_weights.index.to_numpy(),
        size=5,
        replace=False,
        p=(white_weights / white_weights.sum()).to_numpy(),
    )
    pb = int(
        rng.choice(
            pb_weights.index.to_numpy(),
            size=1,
            replace=True,
            p=(pb_weights / pb_weights.sum()).to_numpy(),
        )[0]
    )
    return tuple(sorted(int(x) for x in whites)), pb


def score_ticket(
    ticket_whites: Tuple[int, ...],
    ticket_pb: int,
    actual_row: pd.Series,
    pb_weight: float = 1.5,
) -> Tuple[int, int, float]:
    actual_whites = set(int(x) for x in actual_row[WHITE_COLS].tolist())
    white_hits = len(set(ticket_whites) & actual_whites)
    pb_hit = int(ticket_pb == int(actual_row[PB_COL]))
    total = float(white_hits + pb_hit * pb_weight)
    return white_hits, pb_hit, total


def run_backtest_frequency_recency(
    df: pd.DataFrame,
    split_ratio: float = 0.5,
    recent_n: int = 100,
    alpha: float = 0.7,
    seed: int = 42,
    pb_weight: float = 1.5,
) -> BacktestMetrics:
    train, test = split_by_date(df, split_ratio=split_ratio)
    white_w, pb_w = build_frequency_recency_weights(train, recent_n=recent_n, alpha=alpha)
    rng = np.random.default_rng(seed)

    white_hits: List[int] = []
    pb_hits: List[int] = []
    totals: List[float] = []

    for _, row in test.iterrows():
        t_whites, t_pb = generate_ticket(white_w, pb_w, rng)
        w, p, tot = score_ticket(t_whites, t_pb, row, pb_weight=pb_weight)
        white_hits.append(w)
        pb_hits.append(p)
        totals.append(tot)

    return BacktestMetrics(
        draws_tested=len(test),
        avg_white_matches=float(np.mean(white_hits)) if white_hits else 0.0,
        pb_hit_rate=float(np.mean(pb_hits)) if pb_hits else 0.0,
        avg_total_score=float(np.mean(totals)) if totals else 0.0,
    )


def rank_tickets(
    df_train: pd.DataFrame,
    n_pool: int = 1500,
    recent_n: int = 100,
    alpha: float = 0.7,
    seed: int = 123,
) -> List[Tuple[Tuple[int, ...], int, float]]:
    """
    Produce a ranked list of unique tickets using the same frequency+recency weights.

    Returns: list of (whites_tuple, pb, strength_score) sorted descending.
    """
    white_w, pb_w = build_frequency_recency_weights(df_train, recent_n=recent_n, alpha=alpha)
    rng = np.random.default_rng(seed)

    def strength(whites: Tuple[int, ...], pb: int) -> float:
        return float(sum(white_w.get(w, 0.0) for w in whites) + 0.8 * float(pb_w.get(pb, 0.0)))

    seen = set()
    ranked: List[Tuple[Tuple[int, ...], int, float]] = []

    for _ in range(n_pool):
        whites, pb = generate_ticket(white_w, pb_w, rng)
        key = (whites, pb)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((whites, pb, strength(whites, pb)))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked
