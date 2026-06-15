from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import generate_dataset
from ..utils import DATA_DIR


MODEL_DIR = DATA_DIR / "classifier"


def main():
    data = generate_dataset()
    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=1)),
    ])
    pipe.fit([x["text"] for x in train], [x["label"] for x in train])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "test.json").write_text(json.dumps(test, indent=2))
    print(f"saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()

