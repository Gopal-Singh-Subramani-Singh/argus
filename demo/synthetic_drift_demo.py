"""
Argus synthetic drift demo.
Uses UCI Adult dataset. Injects drift at day 15.
Shows Argus catching it before accuracy drops.

Usage:
  python demo/synthetic_drift_demo.py

Requires:
  pip install scikit-learn pandas requests
  Argus running at http://localhost:8001
"""
from __future__ import annotations
import json
import requests
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

ARGUS_URL = "http://localhost:8001"
MODEL_ID = "adult_income_v1"


def load_adult_dataset():
    print("Loading UCI Adult dataset...")
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult.frame.dropna()
    feature_cols = ["age", "hours-per-week", "education-num",
                    "capital-gain", "capital-loss"]
    X = df[feature_cols].astype(float)
    y = (df["class"] == ">50K").astype(int)
    return X, y, feature_cols


def register_model(feature_cols):
    print(f"Registering model: {MODEL_ID}")
    payload = {
        "model_id": MODEL_ID,
        "name": "Adult Income Classifier",
        "version": "1.0",
        "features": [{"name": col, "type": "numeric"} for col in feature_cols],
    }
    resp = requests.post(f"{ARGUS_URL}/models", json=payload)
    print(f"  Status: {resp.status_code}")
    return resp.json()


def set_reference(X_train, feature_cols):
    print("Setting reference distribution...")
    samples = X_train.head(1000).to_dict(orient="records")
    samples_clean = [{k: float(v) for k, v in row.items()} for row in samples]
    payload = {"model_id": MODEL_ID, "features_data": samples_clean}
    resp = requests.post(f"{ARGUS_URL}/models/{MODEL_ID}/reference", json=payload)
    print(f"  Status: {resp.status_code}")


def simulate_day(X_batch, predictions, model_id, inject_drift=False):
    records = []
    for i in range(len(X_batch)):
        row = X_batch.iloc[i].to_dict()
        if inject_drift:
            row["age"] = float(row["age"]) + 25.0
            row["hours-per-week"] = float(row["hours-per-week"]) * 1.8
        records.append({
            "model_id": model_id,
            "features": {k: float(v) for k, v in row.items()},
            "prediction": int(predictions[i]),
        })
    resp = requests.post(
        f"{ARGUS_URL}/ingest/batch",
        json={"records": records},
    )
    return resp.json()


def trigger_drift_run():
    resp = requests.post(f"{ARGUS_URL}/drift/{MODEL_ID}/run")
    return resp.json()


def main():
    X, y, feature_cols = load_adult_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining GBT classifier...")
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    baseline_acc = model.score(X_test, y_test)
    print(f"Baseline accuracy: {baseline_acc:.3f}")

    try:
        register_model(feature_cols)
        set_reference(X_train, feature_cols)
    except Exception as e:
        print(f"Argus not running: {e}")
        print("Run: uvicorn argus_core.main:app --port 8001")
        return

    print("\nSimulating 20 days of production traffic...")
    print("Drift injected at day 15\n")

    day_size = 100
    for day in range(1, 21):
        inject = day >= 15
        start = (day - 1) * day_size
        end = start + day_size
        X_day = X_test.iloc[start % len(X_test):end % len(X_test)]
        if len(X_day) < day_size:
            X_day = pd.concat([X_day, X_test.iloc[:day_size - len(X_day)]])

        preds = model.predict(X_day)
        result = simulate_day(X_day, preds, MODEL_ID, inject_drift=inject)

        if day % 5 == 0 or inject:
            report = trigger_drift_run()
            severity = report.get("overall_severity", "unknown")
            scores_summary = {
                s["feature_name"]: f"{s['method']}={s['score']:.3f}"
                for s in report.get("scores", [])[:3]
            }
            drift_marker = " ← DRIFT INJECTED" if inject else ""
            print(
                f"Day {day:2d}{drift_marker}: severity={severity} | "
                f"samples={report.get('total_samples', 0)} | "
                f"{scores_summary}"
            )
        else:
            print(f"Day {day:2d}: {result['accepted']} samples ingested")

        time.sleep(0.1)

    print("\nDemo complete.")
    print(f"Check Grafana at http://localhost:3000")
    print(f"Check Argus metrics at {ARGUS_URL}/metrics")


if __name__ == "__main__":
    main()
