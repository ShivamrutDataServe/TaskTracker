"""
Trains the TaskTracker churn-prediction classifier and writes two artifacts:
  - model.pkl     the fitted scikit-learn RandomForestClassifier
  - metrics.json  evaluation metrics + provenance, consumed by:
                     * check_quality_gate.py (CI quality gate)
                     * tests/test_model.py   (regression tests)
                     * app.py                (exposed via /version)

Training happens here, in CI, on its own - never inside `docker build`.
The Docker image only ever COPYs the model.pkl this script produces.
"""
import json
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = "data/churn.csv"
MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "support_calls",
    "is_senior_citizen",
    "has_tech_support",
]
TARGET_COLUMN = "churn"


def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Model trained successfully")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "rows_trained_on": int(len(X_train)),
        "rows_tested_on": int(len(X_test)),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
