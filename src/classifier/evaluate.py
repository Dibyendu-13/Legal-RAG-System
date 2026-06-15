from __future__ import annotations

import json

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..utils import DATA_DIR


def main():
    model = joblib.load(DATA_DIR / "classifier" / "model.joblib")
    test = json.loads((DATA_DIR / "classifier" / "test.json").read_text())
    y_true = [x["label"] for x in test]
    y_pred = model.predict([x["text"] for x in test])
    print("accuracy", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred))
    print("labels", ["billing", "technical_issue", "feature_request", "complaint", "other"])
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
