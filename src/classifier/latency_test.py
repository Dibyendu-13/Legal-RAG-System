from __future__ import annotations

import time

import joblib

from .data import LABELS
from ..utils import DATA_DIR


def main():
    model = joblib.load(DATA_DIR / "classifier" / "model.joblib")
    samples = [
        "I was charged twice for my subscription.",
        "The upload button does nothing.",
        "Please add dark mode.",
        "Your support agent was rude.",
        "What are your hours today?",
    ] * 4
    start = time.perf_counter()
    preds = model.predict(samples)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert all(p in LABELS for p in preds)
    assert elapsed_ms < 500, f"latency too high: {elapsed_ms:.2f}ms"
    print(f"latency_ms={elapsed_ms:.2f}")


if __name__ == "__main__":
    main()

