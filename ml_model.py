#!/usr/bin/env python3
"""
ml_model.py (optional helper)
This shim exists so your code can import a consistent API even if you
swap out internals later.
"""

from typing import List, Set
import random

def pick_unique_numbers(low: int, high: int, k: int) -> List[int]:
    return random.sample(range(low, high+1), k)

def dedupe(picks: List[str], recent: Set[str]) -> List[str]:
    out = []
    for p in picks:
        if p not in recent and p not in out:
            out.append(p)
    return out
