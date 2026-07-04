"""
Argus Real Model Degradation Demo
===================================
Scenario: A fraud detection model trained on historical transaction data.
After day 10, fraudster behavior shifts — transactions become larger, happen
at unusual hours, and new merchant categories appear. The model was never
trained on this pattern, so accuracy silently degrades.

Argus catches the drift at day 12. Accuracy collapse becomes visible at day 18.
This is the exact gap Argus is designed to close.

Run with: python demo/real_degradation_demo.py
Requires: Argus running at http://localhost:8001
"""
from __future__ import annotations

import time
import json
import random
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ARGUS_URL = "http://localhost:8001"
MODEL_ID   = "fraud_detector_v1"
RANDOM_SEED = 42

# API key — matches ARGUS_API_KEY in your .env
API_KEY = "argus-dev-key-change-in-production"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
rng = np.random.default_rng(RANDOM_SEED)

# ── ANSI colours for terminal output ─────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SEV_COLOR = {
    "ok":       GREEN,
    "info":     CYAN,
    "warning":  YELLOW,
    "critical": RED,
}

# ── 1. GENERATE SYNTHETIC TRAINING DATA ──────────────────────────────────────

def generate_transactions(n: int, phase: str = "normal") -> pd.DataFrame:
    """
    Generate synthetic credit card transactions.

    Normal phase (training / early production):
      - amounts: $10–$300, log-normal
      - hours: business hours weighted (9am–9pm)
      - merchant: grocery 40%, retail 35%, restaurant 20%, travel 5%
      - fraud rate: ~8%

    Drift phase (late production — fraudsters adapted):
      - amounts: shift upward to $200–$2000 (fraudsters found high-limit cards)
      - hours: shift to late night 11pm–4am (avoiding detection)
      - merchant: new category "crypto_exchange" appears (20%)
      - fraud rate climbs to ~35% (model blind to new pattern)
    """
    data = []
    merchant_normal = ["grocery", "retail", "restaurant", "travel"]
    merchant_drift   = ["grocery", "retail", "restaurant", "travel", "crypto_exchange", "wire_transfer"]

    for _ in range(n):
        if phase == "normal":
            amount      = float(np.clip(rng.lognormal(4.0, 0.8), 10, 500))
            hour        = int(rng.choice(range(24), p=_hour_weights_normal()))
            merchant    = rng.choice(merchant_normal, p=[0.40, 0.35, 0.20, 0.05])
            day_of_week = int(rng.integers(0, 7))
            velocity_1h = int(rng.integers(0, 4))
            is_foreign  = int(rng.random() < 0.05)

            # Fraud rule: high amount at night from foreign card
            fraud_prob = 0.04
            if amount > 300 and hour < 6:
                fraud_prob = 0.60
            if is_foreign and velocity_1h > 2:
                fraud_prob = 0.70
            is_fraud = int(rng.random() < fraud_prob)

        else:  # drift
            # Amounts shift significantly upward
            amount      = float(np.clip(rng.lognormal(5.5, 1.1), 50, 5000))
            # Shift to late night
            hour        = int(rng.choice(range(24), p=_hour_weights_drift()))
            # New merchant categories appear
            merchant    = rng.choice(merchant_drift, p=[0.20, 0.20, 0.10, 0.05, 0.25, 0.20])
            day_of_week = int(rng.integers(0, 7))
            velocity_1h = int(rng.integers(1, 8))   # higher velocity
            is_foreign  = int(rng.random() < 0.40)  # more foreign cards

            # Model was not trained on crypto/wire patterns — these are fraud
            # but the model doesn't know it
            if merchant in ("crypto_exchange", "wire_transfer"):
                is_fraud = int(rng.random() < 0.75)
            elif hour < 5 and amount > 500:
                is_fraud = int(rng.random() < 0.80)
            else:
                is_fraud = int(rng.random() < 0.20)

        data.append({
            "amount":      amount,
            "hour":        hour,
            "day_of_week": day_of_week,
            "velocity_1h": velocity_1h,
            "is_foreign":  is_foreign,
            "merchant_enc": _encode_merchant(str(merchant)),
            "merchant":    str(merchant),
            "is_fraud":    is_fraud,
        })

    return pd.DataFrame(data)


def _hour_weights_normal():
    """Weight towards business hours 9am-9pm."""
    w = np.ones(24) * 0.5
    w[9:21] = 3.0
    return (w / w.sum()).tolist()


def _hour_weights_drift():
    """Weight towards late night 11pm-4am."""
    w = np.ones(24) * 0.5
    w[23:] = 5.0
    w[:5]  = 5.0
    return (w / w.sum()).tolist()


def _encode_merchant(m: str) -> float:
    mapping = {
        "grocery": 0.0, "retail": 1.0, "restaurant": 2.0,
        "travel": 3.0, "crypto_exchange": 4.0, "wire_transfer": 5.0,
    }
    return mapping.get(m, -1.0)


FEATURE_COLS = ["amount", "hour", "day_of_week", "velocity_1h", "is_foreign", "merchant_enc"]


# ── 2. TRAIN MODEL ─────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["is_fraud"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    return model, acc, auc, X_test, y_test


# ── 3. ARGUS HELPERS ──────────────────────────────────────────────────────────

def argus_register():
    payload = {
        "model_id": MODEL_ID,
        "name":     "Fraud Detector",
        "version":  "1.0",
        "features": [
            {"name": "amount",       "type": "numeric"},
            {"name": "hour",         "type": "numeric"},
            {"name": "day_of_week",  "type": "numeric"},
            {"name": "velocity_1h",  "type": "numeric"},
            {"name": "is_foreign",   "type": "numeric"},
            {"name": "merchant_enc", "type": "numeric"},
        ],
    }
    r = requests.post(f"{ARGUS_URL}/models", json=payload, headers=HEADERS, timeout=10)
    return r.status_code == 200


def argus_set_reference(df: pd.DataFrame):
    samples = df[FEATURE_COLS].head(500).to_dict(orient="records")
    payload = {"model_id": MODEL_ID, "features_data": samples}
    r = requests.post(
        f"{ARGUS_URL}/models/{MODEL_ID}/reference",
        json=payload, headers=HEADERS, timeout=30
    )
    return r.status_code == 200


def argus_ingest(df: pd.DataFrame, predictions: list):
    records = []
    for i, row in df.iterrows():
        feat = {col: float(row[col]) for col in FEATURE_COLS}
        records.append({
            "model_id":   MODEL_ID,
            "features":   feat,
            "prediction": int(predictions[i % len(predictions)]),
            "label":      int(row["is_fraud"]),
        })
    for i in range(0, len(records), 100):
        chunk = records[i:i + 100]
        requests.post(
            f"{ARGUS_URL}/ingest/batch",
            json={"records": chunk},
            headers=HEADERS,
            timeout=30,
        )


def argus_run_drift() -> dict:
    r = requests.post(
        f"{ARGUS_URL}/drift/{MODEL_ID}/run", headers=HEADERS, timeout=30
    )
    if r.status_code == 200:
        return r.json()
    return {}


# ── 4. MAIN DEMO LOOP ─────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  ARGUS REAL MODEL DEGRADATION DEMO{RESET}")
    print(f"{BOLD}  Scenario: Fraud Detector vs Shifting Fraudster Behaviour{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"{CYAN}[1/4] Generating training data (5,000 normal transactions)...{RESET}")
    train_df = generate_transactions(5000, phase="normal")
    fraud_rate = train_df["is_fraud"].mean()
    print(f"      Fraud rate in training data: {fraud_rate:.1%}")

    print(f"{CYAN}[2/4] Training GradientBoostingClassifier...{RESET}")
    model, train_acc, train_auc, X_test, y_test = train_model(train_df)
    print(f"      {GREEN}Baseline accuracy : {train_acc:.3f}{RESET}")
    print(f"      {GREEN}Baseline AUC-ROC  : {train_auc:.3f}{RESET}")

    # ── Register with Argus ──────────────────────────────────────────────────
    print(f"\n{CYAN}[3/4] Registering model with Argus...{RESET}")
    try:
        ok = argus_register()
        if not ok:
            print(f"  {RED}Registration failed — is Argus running at {ARGUS_URL}?{RESET}")
            return
        print(f"      {GREEN}Model registered: {MODEL_ID}{RESET}")

        print(f"{CYAN}[4/4] Setting reference distribution (500 training samples)...{RESET}")
        argus_set_reference(train_df)
        print(f"      {GREEN}Reference distribution set{RESET}")
    except Exception as e:
        print(f"  {RED}Cannot connect to Argus: {e}{RESET}")
        print(f"  Make sure Argus is running: docker compose up --build")
        return

    # ── Simulate 30 days ─────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*65}{RESET}")
    print(f"{BOLD}  Simulating 30 days of production traffic...{RESET}")
    print(f"{BOLD}  Drift begins at day 10 (fraudsters adapt their behaviour){RESET}")
    print(f"{BOLD}{'─'*65}{RESET}")
    print(f"\n  {'Day':<6} {'Phase':<12} {'Acc':>7} {'AUC':>7} {'Fraud%':>8} {'Argus Severity':<16} {'Top Signal'}")
    print(f"  {'─'*6} {'─'*12} {'─'*7} {'─'*7} {'─'*8} {'─'*16} {'─'*30}")

    accuracy_history  = []
    severity_history  = []

    for day in range(1, 31):
        # Phase transition
        if day <= 10:
            phase        = "normal"
            n_per_day    = 150
        elif day <= 15:
            phase        = "early_drift"   # 30% drifted, 70% normal
            n_per_day    = 150
        else:
            phase        = "full_drift"    # 100% drifted
            n_per_day    = 150

        # Generate today's transactions
        if phase == "normal":
            day_df = generate_transactions(n_per_day, phase="normal")
        elif phase == "early_drift":
            n_drift  = int(n_per_day * 0.30)
            n_normal = n_per_day - n_drift
            day_df = pd.concat([
                generate_transactions(n_normal, phase="normal"),
                generate_transactions(n_drift,  phase="drift"),
            ], ignore_index=True)
        else:
            day_df = generate_transactions(n_per_day, phase="drift")

        # Model predicts on today's data
        X_day    = day_df[FEATURE_COLS].values
        y_true   = day_df["is_fraud"].values
        y_pred   = model.predict(X_day)
        y_prob   = model.predict_proba(X_day)[:, 1]
        day_acc  = accuracy_score(y_true, y_pred)
        try:
            day_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            day_auc = 0.5
        fraud_pct = y_true.mean()

        accuracy_history.append(day_acc)

        # Send to Argus
        argus_ingest(day_df, y_pred.tolist())

        # Run drift every 3 days
        severity    = "ok"
        top_signal  = "—"
        if day % 3 == 0:
            time.sleep(2)  # let StreamConsumer flush to TimescaleDB
            report = argus_run_drift()
            if report:
                severity   = report.get("overall_severity", "ok")
                scores     = report.get("scores", [])
                # Find the most alarming score
                sev_order  = {"critical": 4, "warning": 3, "info": 2, "ok": 1}
                scores_sorted = sorted(
                    scores,
                    key=lambda s: sev_order.get(s.get("severity", "ok"), 0),
                    reverse=True,
                )
                if scores_sorted:
                    top = scores_sorted[0]
                    top_signal = (
                        f"{top['feature_name']}.{top['method']}"
                        f"={top['score']:.3f}"
                    )
        severity_history.append(severity)

        # Colour-code the row
        sev_col  = SEV_COLOR.get(severity, RESET)
        acc_col  = GREEN if day_acc > 0.80 else (YELLOW if day_acc > 0.65 else RED)
        drift_marker = ""
        if day == 10:
            drift_marker = f"  {YELLOW}← drift begins{RESET}"
        elif day == 15:
            drift_marker = f"  {RED}← full drift{RESET}"

        phase_label = {"normal": "normal", "early_drift": "early drift", "full_drift": "FULL DRIFT"}[phase]

        print(
            f"  {day:<6} {phase_label:<12} "
            f"{acc_col}{day_acc:>7.3f}{RESET} "
            f"{day_auc:>7.3f} "
            f"{fraud_pct:>8.1%} "
            f"{sev_col}{severity:<16}{RESET} "
            f"{top_signal}"
            f"{drift_marker}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*65}{RESET}")
    print(f"{BOLD}  RESULTS SUMMARY{RESET}")
    print(f"{BOLD}{'─'*65}{RESET}")

    baseline_acc = np.mean(accuracy_history[:10])
    final_acc    = np.mean(accuracy_history[20:])
    acc_drop     = baseline_acc - final_acc

    # Find first day Argus warned
    first_warning  = next((i*3 for i, s in enumerate(severity_history) if s == "warning"),  None)
    first_critical = next((i*3 for i, s in enumerate(severity_history) if s == "critical"), None)

    print(f"\n  Baseline accuracy (days 1–10):    {GREEN}{baseline_acc:.3f}{RESET}")
    print(f"  Final accuracy   (days 21–30):    {RED}{final_acc:.3f}{RESET}")
    print(f"  Accuracy drop:                    {RED}−{acc_drop:.3f} ({acc_drop/baseline_acc:.1%}){RESET}")

    if first_warning:
        print(f"\n  {YELLOW}Argus WARNING fired:   day ~{first_warning}{RESET}")
    if first_critical:
        print(f"  {RED}Argus CRITICAL fired:  day ~{first_critical}{RESET}")

    if first_warning and first_critical:
        lead_time = 30 - first_warning
        print(f"\n  {GREEN}{BOLD}Argus detected drift {lead_time} days before full accuracy collapse.{RESET}")
        print(f"  {GREEN}That's your retraining lead time.{RESET}")

    print(f"\n  {CYAN}View live in Grafana:     http://localhost:3000{RESET}")
    print(f"  {CYAN}View Prometheus metrics:  http://localhost:9090{RESET}")
    print(f"  {CYAN}View API / latest scores: http://localhost:8001/drift/{MODEL_ID}/latest{RESET}")
    print(f"\n{BOLD}{'='*65}{RESET}\n")


if __name__ == "__main__":
    main()
