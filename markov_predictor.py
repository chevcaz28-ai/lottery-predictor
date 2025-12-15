import os
import random

def _init_rng():
    """Initialize RNG safely.
    Accepts MARKOV_RANDOM_SEED=int or 'auto'/empty.
    """
    seed_raw = os.getenv("MARKOV_RANDOM_SEED", "123")
    seed_raw = seed_raw.strip().lower()

    if seed_raw in ("auto", "", "none", "null"):
        return random.Random()

    try:
        return random.Random(int(seed_raw))
    except ValueError:
        print(f"[WARN] Invalid MARKOV_RANDOM_SEED='{seed_raw}', falling back to auto seed")
        return random.Random()


def run():
    rng = _init_rng()
    print("[MARKOV] RNG initialized")
    # ---- existing Markov logic continues below unchanged ----


if __name__ == "__main__":
    run()
