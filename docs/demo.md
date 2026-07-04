# Argus — Demo Guide

## What this demo proves

- Model registration and feature schema definition
- Prediction ingestion via REST API and SDK
- Drift computation using KS test, PSI, and Jensen-Shannon divergence
- Alert rules evaluation against configurable thresholds
- Prometheus metrics populated and queryable
- Grafana drift dashboard live

---

## Prerequisites

```bash
pip install -r requirements.txt
docker compose up timescaledb redis prometheus grafana -d
cp .env.example .env
```

---

## Demo Commands

### 1. Start Argus

```bash
uvicorn argus_core.main:app --port 8001 --reload
```

### 2. Verify health

```bash
curl http://localhost:8001/health
```

Expected:

```json
{"status": "healthy", "redis": "connected", "db": "connected"}
```

### 3. Register a model

```bash
curl -X POST http://localhost:8001/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud_v1",
    "name": "Fraud Detector",
    "version": "1.0",
    "features": [
      {"name": "amount", "type": "numeric"},
      {"name": "merchant_type", "type": "categorical"}
    ]
  }'
```

### 4. Run the synthetic drift simulation

```bash
python demo/synthetic_drift_demo.py
```

This script:
- Ingests 100 predictions with a stable distribution (reference)
- Sets the reference distribution
- Ingests 100 predictions with a shifted distribution (drift)
- Triggers drift computation
- Prints computed drift scores

Expected output:
```
[argus] Reference distribution set for fraud_v1
[argus] Ingested 100 stable predictions
[argus] Ingested 100 drifted predictions
[argus] Drift triggered → KS p-value: 0.02 (CRITICAL), PSI: 0.31 (CRITICAL)
[argus] Alert fired: psi_critical → retraining_trigger=True
```

### 5. Manually trigger drift computation

```bash
curl -X POST http://localhost:8001/drift/fraud_v1/run
```

### 6. View latest drift scores

```bash
curl http://localhost:8001/drift/fraud_v1/latest | python -m json.tool
```

### 7. View Prometheus metrics

```bash
curl http://localhost:8001/metrics | grep argus_
```

### 8. View Grafana dashboard

```
http://localhost:3000   (admin / argus)
Import: dashboards/argus_grafana.json
```

Screenshot pending.

### 9. Interactive API docs

```
http://localhost:8001/docs
```

---

## Expected Output Summary

| Check | Expected |
|-------|----------|
| `/health` | Redis and DB connected |
| Model registration | 200 OK, model stored |
| Drift demo | KS and PSI scores computed, alerts fired |
| `/drift/{id}/latest` | Drift scores per feature per method |
| `/metrics` | argus_drift_score, argus_alerts_fired_total populated |
| Grafana | Drift score time series, alert firing events |

---

## Known Limitations

- Requires TimescaleDB running (via docker compose). Without it, ingestion succeeds but drift score storage fails.
- SHAP drift detection may not be fully complete — verify `argus_core/drift/shap_drift.py` before demoing.
- Alert deduplication is not implemented — repeated threshold breaches fire repeated alerts.
