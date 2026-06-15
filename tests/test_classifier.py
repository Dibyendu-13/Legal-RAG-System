from __future__ import annotations

import subprocess
import sys
import time

import joblib

from src.classifier.data import LABELS
from src.utils import DATA_DIR


def test_classifier_outputs_valid_labels():
    model = joblib.load(DATA_DIR / "classifier" / "model.joblib")
    preds = model.predict(
        [
            "I was charged twice this month.",
            "The upload button is broken.",
            "Please add dark mode.",
            "Support was rude and dismissive.",
            "What are your business hours?",
        ]
    )
    assert all(p in LABELS for p in preds)


def test_classifier_latency_under_500ms():
    model = joblib.load(DATA_DIR / "classifier" / "model.joblib")
    samples = ["I was charged twice for the subscription."] * 20
    start = time.perf_counter()
    model.predict(samples)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500


def test_classifier_evaluate_script_runs():
    out = subprocess.run(
        [sys.executable, "-m", "src.classifier.evaluate"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "accuracy" in out.stdout
    assert "confusion matrix" not in out.stdout.lower()

