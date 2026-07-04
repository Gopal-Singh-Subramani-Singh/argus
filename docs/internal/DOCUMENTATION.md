# Argus — ML Observability & Drift Detection Platform

> **Argus watches your deployed models so you don't have to.** It detects silent degradation before users notice it, fires alerts before accuracy collapses, and gives you the lead time to retrain.

---

## Table of Contents

1. [What Is Argus](#1-what-is-argus)
2. [The Core Problem](#2-the-core-problem)
3. [Architecture](#3-architecture)
4. [The Five Drift Methods](#4-the-five-drift-methods)
5. [Project Structure](#5-project-structure)
6. [Security](#6-security)
7. [How to Run](#7-how-to-run)
8. [SDK Usage](#8-sdk-usage)
9. [API Reference](#9-api-reference)
10. [Alert Rules](#10-alert-rules)
11. [Prometheus Metrics](#11-prometheus-metrics)
12. [Grafana Dashboard](#12-grafana-dashboard)
13. [Configuration Reference](#13-configuration-reference)
14. [Port Reference](#14-port-reference)
15. [Running Tests](#15-running-tests)
16. [Demo Scenarios](#16-demo-scenarios)
17. [Production Hardening](#17-production-hardening)

---

## 1. What Is Argus

Argus is a self-hosted ML observability platform that monitors deployed machine learning models for **data drift** — the silent, gradual shift between the data a model was trained on and the data it sees in production.


**What Argus does:**

- Ingests feature vectors and predictions from any model via a REST API and a non-blocking Python SDK
- Buffers all traffic through Redis Streams so your inference latency is never affected
- Runs five drift detection algorithms on a schedule (every 5 minutes by default)
- Stores all drift scores as a time-series in TimescaleDB for historical analysis
- Evaluates YAML-defined alert rules and fires webhooks — including retraining triggers — when thresholds are crossed
- Exposes 15 Prometheus metrics and an auto-provisioned Grafana dashboard
- Provides a DuckDB analytics layer for 30-day drift history queries

**What Argus does not do:**

- It does not retrain models itself. It fires a webhook that you wire to your retraining pipeline.
- It does not serve predictions. It only observes them.
- It does not require changes to your model's inference code beyond a two-line SDK integration.

---

## 2. The Core Problem

A model trained on historical data is deployed to production. Over time, the world changes:

- User behavior shifts
- New customer segments appear
- Upstream data pipelines change encoding
- Seasonal patterns evolve

The model's accuracy degrades **silently**. There are no errors, no exceptions, no alerts from your infrastructure — just slowly worsening predictions. By the time a business metric signals a problem, weeks of bad predictions have already shipped.


**The Argus demo proves this concretely:** a fraud detection model trained on normal transaction patterns experiences a fraudster behavior shift at day 10. Model accuracy starts declining. Argus detects distribution drift at day 12 and fires a critical alert. Full accuracy collapse doesn't become visible until day 18 — giving an operations team a **6-day lead time** to retrain before the model causes significant business harm.

Argus catches the problem at the feature distribution level, not at the accuracy level, which is what makes early detection possible. You do not need ground truth labels to detect drift.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your ML Service                             │
│                                                                     │
│   model.predict(features)                                           │
│   argus.log(features, prediction)  ◄── 2-line SDK integration      │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ HTTP POST /ingest/batch (non-blocking, batched)
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Argus API (FastAPI / uvicorn)                  │
│                         Port 8001                                   │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
│  │  /models    │  │  /ingest     │  │  /drift    │  │ /metrics │  │
│  │  registry   │  │  batch       │  │  run/query │  │ /health  │  │
│  └─────────────┘  └──────┬───────┘  └─────┬──────┘  └──────────┘  │
└─────────────────────────┼──────────────────┼────────────────────────┘
                          │                  │
                          ▼                  │
              ┌───────────────────┐          │
              │   Redis Streams   │          │
              │   Port 6380       │          │
              │   argus:ingest:   │          │
              │   stream          │          │
              └────────┬──────────┘          │
                       │                     │
              ┌────────▼──────────┐          │
              │  StreamConsumer   │          │
              │  (async loop)     │          │
              └────────┬──────────┘          │
                       │                     │
                       ▼                     ▼
              ┌─────────────────────────────────────────┐
              │           TimescaleDB  Port 5432         │
              │                                          │
              │  feature_logs          (raw vectors)     │
              │  drift_scores          (time-series)     │
              │  reference_distributions                 │
              │  alert_history                           │
              │  models                                  │
              └───────────────────┬─────────────────────┘
                                  │
              ┌───────────────────▼─────────────────────┐
              │           Drift Engine (APScheduler)     │
              │           Every 300s (configurable)      │
              │                                          │
              │  KS Test ── PSI ── Chi² ── JSD ── SHAP  │
              └───────────────────┬─────────────────────┘
                                  │
              ┌───────────────────▼─────────────────────┐
              │           Alert Engine                   │
              │           Evaluates alert_rules.yaml     │
              │           Fires webhooks with retry      │
              └───────────────────┬─────────────────────┘
                                  │
              ┌───────────────────▼─────────────────────┐
              │  Prometheus (Port 9090) + Grafana (3000) │
              │  15 metrics, auto-provisioned dashboard  │
              └─────────────────────────────────────────┘
```


**Data flow summary:**

1. Your model calls `argus.log()` after each prediction — returns immediately, never blocks
2. The SDK batches records and posts them to `/ingest/batch` every 100ms
3. The Ingest Service writes each record to a Redis Stream (`argus:ingest:stream`) with a 100k message cap
4. The StreamConsumer reads from the stream and persists rows to the `feature_logs` hypertable in TimescaleDB
5. The DriftEngine runs every 5 minutes (APScheduler), fetches the last 24 hours of production data and the reference distribution, computes all 5 drift methods, and persists scores to `drift_scores`
6. The AlertEngine evaluates `alert_rules.yaml` against the new scores and fires webhooks with up to 3 retries
7. Prometheus scrapes `/metrics` every 15 seconds; Grafana displays everything on an auto-provisioned dashboard

---

## 4. The Five Drift Methods

Argus runs five drift detection algorithms. They complement each other: statistical tests, information-theoretic measures, and model-level behavioral drift.

### 4.1 Kolmogorov-Smirnov Test (KS Test)

**Applies to:** Numeric features

**What it measures:** The maximum vertical distance between the empirical cumulative distribution functions (CDFs) of the reference and production distributions.

**Interpretation:** The test returns a statistic (0–1, where 0 means identical distributions) and a p-value. A low p-value means the distributions are statistically different. Argus uses the p-value for severity classification.

**Thresholds:**

| p-value | Severity |
|---------|----------|
| > 0.20  | `ok`     |
| 0.10 – 0.20 | `info` |
| 0.05 – 0.10 | `warning` |
| < 0.05  | `critical` |

**When to use:** Best for detecting shifts in the shape of continuous feature distributions. Very sensitive to any distributional change. The KS test is non-parametric and makes no assumptions about the underlying distribution.

---

### 4.2 Population Stability Index (PSI)

**Applies to:** Numeric features (auto-binned with percentile-based edges) and categorical features

**What it measures:** PSI = Σ (actual% − expected%) × ln(actual% / expected%) across all bins. Measures how much the population distribution has shifted from a reference snapshot.

**Interpretation:** PSI is additive across bins. It penalizes both the magnitude and the asymmetry of the shift.

**Thresholds:**

| PSI score | Severity |
|-----------|----------|
| < 0.05    | `ok`     |
| 0.05 – 0.10 | `info` |
| 0.10 – 0.25 | `warning` |
| > 0.25    | `critical` |

**When to use:** PSI is the industry standard for monitoring credit and fraud models. It is interpretable, stable, and works for both numeric and categorical features. Use it when you want a single number that summarizes population shift.


---

### 4.3 Chi-Squared Test

**Applies to:** Categorical features

**What it measures:** The chi-squared statistic comparing observed category frequencies in production versus expected frequencies from the reference distribution. Tests whether the two categorical distributions are drawn from the same population.

**Thresholds:**

| p-value | Severity |
|---------|----------|
| > 0.10  | `ok`     |
| 0.05 – 0.10 | `warning` |
| < 0.05  | `critical` |

**When to use:** The natural choice for categorical features like merchant category, country code, device type, or any enumerated value. The KS test cannot be applied to categoricals — chi-squared is the correct tool.

---

### 4.4 Jensen-Shannon Divergence (JSD)

**Applies to:** Numeric features

**What it measures:** A symmetrized and smoothed version of KL divergence. JSD(P || Q) = 0.5 × KL(P || M) + 0.5 × KL(Q || M) where M = 0.5(P + Q). Measures how different two probability distributions are.

**Interpretation:** JSD is bounded between 0 (identical distributions) and 1 (completely different), making it more intuitive than raw KL divergence. It is symmetric — JSD(P,Q) = JSD(Q,P).

**Thresholds:**

| JSD score | Severity |
|-----------|----------|
| < 0.05    | `ok`     |
| 0.05 – 0.10 | `info` |
| 0.10 – 0.30 | `warning` |
| > 0.30    | `critical` |

**When to use:** JSD catches distributional divergence even when both distributions have similar means and variances (shape changes that KS and PSI might miss). Good as a complement to KS test for important numeric features.

---

### 4.5 SHAP Drift (Feature Importance Rank Correlation)

**Applies to:** The model as a whole (all numeric features combined)

**What it measures:** Instead of per-feature distribution shift, SHAP drift measures whether the **relative importance of features** has changed. It computes mean absolute SHAP values (or a variance-based proxy when no model artifact is provided) per feature for both reference and production, then computes the Spearman rank correlation between the two importance vectors.

**Why rank correlation:** Raw SHAP magnitudes are unstable across sample sets. Rank order of feature importance is much more robust. A change in rank order means the model is relying on different signals — even if individual feature distributions look stable.

**Interpretation:** A rank correlation close to 1.0 means feature importance is stable. A low correlation (< 0.7) means the model is now driven by different features than during training.

**Thresholds:**

| Rank correlation | Severity |
|-----------------|----------|
| > 0.85          | `ok`     |
| 0.70 – 0.85     | `info`   |
| 0.50 – 0.70     | `warning` |
| < 0.50          | `critical` |

**When to use:** SHAP drift is the highest-signal indicator. It catches cases where features themselves look stable but the model's reliance on them has shifted — an early indicator of concept drift. Requires at least 2 numeric features.

---

### Drift Method Summary

| Method | Feature Type | Detects | Reported As |
|--------|-------------|---------|-------------|
| KS Test | Numeric | Shape/location shift | statistic + p-value |
| PSI | Numeric + Categorical | Population shift | PSI score |
| Chi-Squared | Categorical | Category frequency shift | statistic + p-value |
| JS Divergence | Numeric | Distributional divergence | JSD score (0–1) |
| SHAP Drift | Model-level (numeric) | Feature importance reordering | Rank correlation (0–1) |


---

## 5. Project Structure

```
argus/
├── argus_core/                     # Core application package
│   ├── main.py                     # FastAPI app, auth middleware, all route handlers
│   ├── ingest.py                   # IngestService: validates and writes to Redis Streams
│   ├── consumer.py                 # StreamConsumer: reads Redis → writes TimescaleDB
│   ├── models.py                   # All Pydantic request/response models
│   ├── metrics.py                  # 15 Prometheus metric definitions
│   ├── drift/
│   │   ├── engine.py               # APScheduler-based drift scheduler + orchestrator
│   │   ├── ks_test.py              # Kolmogorov-Smirnov test (scipy.stats.ks_2samp)
│   │   ├── psi.py                  # Population Stability Index (numeric + categorical)
│   │   ├── chi_squared.py          # Chi-squared test for categorical features
│   │   ├── js_divergence.py        # Jensen-Shannon divergence
│   │   └── shap_drift.py           # SHAP rank correlation via Spearman
│   ├── store/
│   │   ├── timescale.py            # AsyncPG TimescaleDB client + schema helpers
│   │   └── duckdb_analytics.py     # DuckDB in-memory analytics for 30-day history
│   ├── alerts/
│   │   ├── engine.py               # YAML rule evaluator, fires alerts on threshold breach
│   │   └── webhook.py              # Webhook sender with tenacity retry (3 attempts)
│   └── registry/
│       └── model_registry.py       # In-memory + DB model registry, reference store
├── config/
│   ├── settings.py                 # Pydantic-based config, env var injection
│   ├── config.yaml                 # Application configuration (non-secret)
│   └── alert_rules.yaml            # Alert rule definitions (edit to customize)
├── sdk/
│   ├── __init__.py                 # Exports argus.init() and argus.log()
│   └── client.py                   # ArgusClient: threaded, non-blocking, batched SDK
├── tests/                          # 40 pytest tests (all mocked, no Docker needed)
│   ├── conftest.py                 # Shared fixtures
│   ├── test_alert_engine.py
│   ├── test_chi_squared.py
│   ├── test_drift_engine.py
│   ├── test_ingest.py
│   ├── test_js_divergence.py
│   ├── test_ks_test.py
│   ├── test_psi.py
│   └── test_shap_drift.py
├── demo/
│   ├── real_degradation_demo.py    # Fraud model degradation simulation (30-day)
│   └── synthetic_drift_demo.py     # Simple synthetic drift example
├── scripts/
│   └── init_timescaledb.sql        # TimescaleDB schema: hypertables, indexes
├── grafana/
│   └── provisioning/               # Auto-provisioned datasource + dashboard
├── dashboards/
│   └── argus.json                  # Grafana dashboard JSON
├── .env.example                    # Secrets template — copy to .env before running
├── .gitignore
├── docker-compose.yml              # Full stack: argus, timescaledb, redis, prometheus, grafana
├── Dockerfile                      # Non-root user argus (UID 1001), python:3.11-slim
├── prometheus.yml                  # Prometheus scrape config (scrapes argus:8001/metrics)
├── pyproject.toml
└── requirements.txt                # All pinned dependencies
```

---

## 6. Security

### API Key Authentication

All API endpoints require an `X-API-Key` header except:

- `GET /` — root info
- `GET /health` — health check
- `GET /metrics` — Prometheus scrape endpoint
- `GET /docs` — Swagger UI
- `GET /openapi.json` — OpenAPI schema

The API key is read from the `ARGUS_API_KEY` environment variable at startup. If `ARGUS_API_KEY` is not set, authentication is disabled (development mode). In production, always set this.

```bash
# All authenticated requests
curl -H "X-API-Key: your-secret-key" http://localhost:8001/models
```

A rejected request returns:

```json
HTTP 401
{
  "detail": "Invalid or missing API key. Include header: X-API-Key: <key>"
}
```

All rejected requests are logged with the IP address for audit purposes.


### Environment Variables and Secrets

All secrets are injected via environment variables. The `.env.example` file documents every variable:

```env
# API authentication key (required in production)
ARGUS_API_KEY=change-me-to-a-long-random-secret

# TimescaleDB credentials
POSTGRES_USER=argus
POSTGRES_PASSWORD=change-me-strong-db-password
POSTGRES_DB=argus

# Grafana admin credentials
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=change-me-grafana-password

# Redis password (recommended in production)
REDIS_PASSWORD=change-me-redis-password

# Uvicorn worker count (set to CPU count in production)
ARGUS_WORKERS=2

# Log verbosity: debug | info | warning | error
LOG_LEVEL=info
```

**Important rules:**

- Never commit `.env` to version control. It is listed in `.gitignore`.
- Use a cryptographically random value for `ARGUS_API_KEY` (e.g., `openssl rand -hex 32`)
- Rotate credentials by updating `.env` and restarting the stack

### Container Security

The Docker container runs as a non-root user `argus` (UID 1001, GID 1001). This is enforced in the Dockerfile:

```dockerfile
RUN groupadd --gid 1001 argus && \
    useradd --uid 1001 --gid argus --shell /bin/bash --create-home argus
USER argus
```

The `config/` directory is mounted read-only into the container (`./config:/app/config:ro`).

---

## 7. How to Run

### Prerequisites

- Docker Engine 24+ and Docker Compose v2
- 4 GB RAM available (TimescaleDB + Redis + Argus)
- Ports 8001, 5432, 6380, 9090, and 3000 available on your machine

### Step 1 — Clone and configure secrets

```bash
cd argus/
cp .env.example .env
```

Edit `.env` and replace all `change-me-*` placeholders with real values. At minimum:

```env
ARGUS_API_KEY=<generate with: openssl rand -hex 32>
POSTGRES_PASSWORD=<strong password>
REDIS_PASSWORD=<strong password>
GF_SECURITY_ADMIN_PASSWORD=<grafana password>
```

### Step 2 — Start the full stack

```bash
docker compose up --build
```

This starts five containers in dependency order:

1. `timescaledb` — PostgreSQL 15 with TimescaleDB extension, initializes schema from `scripts/init_timescaledb.sql`
2. `redis` — Redis 7 Alpine with password auth, AOF persistence, 512MB memory limit
3. `argus` — FastAPI server (waits for timescaledb and redis health checks to pass)
4. `prometheus` — Scrapes `argus:8001/metrics` every 15s, 30-day retention
5. `grafana` — Auto-provisions Prometheus datasource and Argus dashboard

First start takes 60–90 seconds while TimescaleDB initializes. Watch for:

```
argus_server  | {"event": "argus.started", "api_key_enabled": true}
```

### Step 3 — Verify the stack is healthy

```bash
curl http://localhost:8001/health
```

Expected response when all services are connected:

```json
{
  "status": "ok",
  "timescaledb": "ok",
  "redis": "ok",
  "uptime_seconds": 12.4
}
```

### Step 4 — Open the interfaces

| Interface | URL | Credentials |
|-----------|-----|-------------|
| Argus API docs | http://localhost:8001/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / GF_SECURITY_ADMIN_PASSWORD |

### Step 5 — Register a model and start ingesting

```bash
API_KEY="your-key-from-.env"

# Register a model
curl -X POST http://localhost:8001/models \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my_model_v1",
    "name": "My Model",
    "version": "1.0",
    "features": [
      {"name": "feature_a", "type": "numeric"},
      {"name": "feature_b", "type": "categorical"}
    ]
  }'

# Set reference distribution
curl -X POST http://localhost:8001/models/my_model_v1/reference \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my_model_v1",
    "features_data": [
      {"feature_a": 1.2, "feature_b": "cat_A"},
      {"feature_a": 2.4, "feature_b": "cat_B"}
    ]
  }'
```

### Stopping the stack

```bash
docker compose down          # Stop containers, preserve data volumes
docker compose down -v       # Stop containers and delete all data volumes
```


---

## 8. SDK Usage

The Python SDK provides a non-blocking, thread-safe interface to Argus. Your inference service calls `argus.log()` immediately after `model.predict()`. The SDK batches records in a background thread and flushes them every 100ms. If Argus is unreachable, the SDK fails silently — it never blocks or raises an exception in your inference path.

### Installation

The SDK is in `sdk/client.py`. Copy it into your project or install the package:

```bash
pip install httpx structlog   # SDK dependencies
```

### Basic Usage

```python
import argus

# Call once at application startup
argus.init(
    endpoint="http://localhost:8001",
    model_id="my_model_v1",
    flush_interval_ms=100,   # How often to batch-send, default 100ms
)

# Call after every prediction — non-blocking, thread-safe
def predict(features: dict) -> float:
    prediction = model.predict(features)
    argus.log(
        features=features,
        prediction=prediction,
        label=None,       # Optional: set when ground truth is available
        metadata={"request_id": "abc123"},
    )
    return prediction
```

### Advanced: Direct Client Instance

If you need multiple models or more control:

```python
from sdk.client import ArgusClient

client = ArgusClient(
    endpoint="http://argus:8001",
    model_id="fraud_detector_v2",
    flush_interval_ms=100,
    max_batch_size=50,       # Max records per HTTP request
    timeout_seconds=5,       # HTTP timeout — client drops silently on timeout
).start()

# In your prediction handler
client.log(features={"amount": 150.0, "hour": 14}, prediction=0)

# On application shutdown
client.stop()
```

### SDK Design Guarantees

| Property | Behavior |
|----------|---------|
| **Non-blocking** | `argus.log()` puts records on an in-process queue and returns immediately |
| **Thread-safe** | Uses `queue.Queue` for lock-free producer/consumer |
| **Fail-silent** | Network errors, timeouts, and HTTP failures are logged at DEBUG level only |
| **Backpressure** | When the queue is full, records are silently dropped (never blocks) |
| **Batched** | Records are grouped into batches of up to 50 before sending |
| **Daemon thread** | The flush thread is a daemon — it does not prevent process exit |

### Adding Labels After the Fact

Ground truth labels often arrive hours or days after predictions. Log them separately when available:

```python
# When you receive actual outcome
argus.log(
    features=original_features,
    prediction=original_prediction,
    label=actual_outcome,       # Ground truth
    metadata={"delayed_label": True, "original_request_id": "abc123"},
)
```

---

## 9. API Reference

All endpoints except `/`, `/health`, `/metrics`, `/docs`, and `/openapi.json` require the `X-API-Key` header.

Interactive documentation is available at **http://localhost:8001/docs** (Swagger UI) and **http://localhost:8001/redoc**.

### Model Registry

#### `POST /models` — Register a model

Registers a new model and its feature schema. The schema defines feature names and types, which Argus uses to apply the correct drift methods per feature.

**Request body:**

```json
{
  "model_id": "fraud_detector_v1",
  "name": "Fraud Detector",
  "version": "1.0",
  "features": [
    {"name": "amount",      "type": "numeric",      "description": "Transaction amount USD"},
    {"name": "hour",        "type": "numeric",      "description": "Hour of day 0-23"},
    {"name": "merchant",    "type": "categorical",  "description": "Merchant category"}
  ],
  "metadata": {"team": "risk", "environment": "production"}
}
```

**Response `200`:**

```json
{
  "model_id": "fraud_detector_v1",
  "name": "Fraud Detector",
  "version": "1.0",
  "registered_at": "2024-01-15T10:30:00",
  "status": "registered"
}
```


---

#### `POST /models/{model_id}/reference` — Set reference distribution

Sets the reference (training) distribution for a model. Argus computes drift by comparing production data against this reference. Call this once after registration using a representative sample from your training set (200–1000 samples recommended).

**Request body:**

```json
{
  "model_id": "fraud_detector_v1",
  "features_data": [
    {"amount": 120.5, "hour": 14, "merchant": "grocery"},
    {"amount": 45.0,  "hour": 9,  "merchant": "retail"}
  ]
}
```

**Response `200`:**

```json
{
  "model_id": "fraud_detector_v1",
  "status": "reference_set",
  "features": 3
}
```

---

#### `POST /models/{model_id}/reference/refresh` — Refresh reference from production data

Re-computes the reference distribution from the last 30 days of production logs. Useful after retraining or when the reference has drifted intentionally. Requires at least 50 logged samples.

**Response `200`:**

```json
{
  "model_id": "fraud_detector_v1",
  "status": "reference_refreshed",
  "features_updated": 3,
  "samples_used": 1240
}
```

---

#### `GET /models` — List all registered models

**Response `200`:** Array of model objects.

#### `GET /models/{model_id}` — Get a specific model

**Response `200`:** Model object with feature schema.

#### `DELETE /models/{model_id}` — Delete a model

**Response `200`:**

```json
{"model_id": "fraud_detector_v1", "status": "deleted"}
```

---

### Ingestion

#### `POST /ingest` — Ingest a single prediction

```json
{
  "model_id": "fraud_detector_v1",
  "features": {"amount": 150.0, "hour": 22, "merchant": "crypto_exchange"},
  "prediction": 0,
  "label": null,
  "metadata": {"user_id": "u_123"}
}
```

**Response `200`:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted",
  "queued": true
}
```

---

#### `POST /ingest/batch` — Ingest a batch of predictions

Preferred for production use. The SDK uses this endpoint automatically.

```json
{
  "records": [
    {
      "model_id": "fraud_detector_v1",
      "features": {"amount": 150.0, "hour": 22, "merchant": "grocery"},
      "prediction": 0
    },
    {
      "model_id": "fraud_detector_v1",
      "features": {"amount": 3200.0, "hour": 2, "merchant": "crypto_exchange"},
      "prediction": 1
    }
  ]
}
```

**Response `200`:**

```json
{
  "accepted": 2,
  "rejected": 0,
  "total": 2
}
```

A record is rejected if the `model_id` has not been registered.

---

### Drift

#### `POST /drift/{model_id}/run` — Trigger drift computation immediately

Bypasses the scheduler and runs all drift methods right now. Returns the full drift report.

**Response `200`:**

```json
{
  "model_id": "fraud_detector_v1",
  "window_start": "2024-01-15T09:00:00",
  "window_end": "2024-01-15T10:00:00",
  "total_samples": 342,
  "overall_severity": "warning",
  "generated_at": "2024-01-15T10:00:01",
  "scores": [
    {
      "feature_name": "amount",
      "method": "ks_test",
      "score": 0.312,
      "p_value": 0.003,
      "severity": "critical",
      "sample_count": 342,
      "scored_at": "2024-01-15T10:00:01"
    },
    {
      "feature_name": "amount",
      "method": "psi",
      "score": 0.18,
      "severity": "warning",
      "sample_count": 342,
      "scored_at": "2024-01-15T10:00:01"
    },
    {
      "feature_name": "_model",
      "method": "shap_drift",
      "score": 0.62,
      "severity": "warning",
      "sample_count": 342,
      "scored_at": "2024-01-15T10:00:01"
    }
  ]
}
```


---

#### `GET /drift/{model_id}/latest` — Latest drift scores

Returns the 100 most recent drift scores for a model (all features, all methods).

**Response `200`:** Array of score objects sorted by `scored_at` descending.

---

#### `GET /drift/{model_id}/history` — 30-day drift time-series

Returns the drift history for a specific feature and method combination.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature` | string | required | Feature name |
| `method` | string | `psi` | Drift method: `ks_test`, `psi`, `chi_squared`, `js_divergence`, `shap_drift` |
| `days` | integer | `30` | Number of days of history |

**Example:**

```bash
curl "http://localhost:8001/drift/fraud_detector_v1/history?feature=amount&method=psi&days=14" \
  -H "X-API-Key: $API_KEY"
```

**Response `200`:**

```json
{
  "model_id": "fraud_detector_v1",
  "feature_name": "amount",
  "method": "psi",
  "days": 14,
  "history": [
    {"scored_at": "2024-01-01T10:00:00", "score": 0.02, "severity": "ok", "sample_count": 310},
    {"scored_at": "2024-01-04T10:00:00", "score": 0.08, "severity": "info", "sample_count": 298},
    {"scored_at": "2024-01-07T10:00:00", "score": 0.19, "severity": "warning", "sample_count": 334}
  ]
}
```

---

#### `GET /alerts/{model_id}/history` — Alert history

**Query parameters:** `days` (default: 7)

**Response `200`:** Array of alert records including rule name, scores, thresholds, and whether webhooks fired.

---

### Observability

#### `GET /health` — Health check

No authentication required. Returns status of all dependencies.

```json
{
  "status": "ok",
  "timescaledb": "ok",
  "redis": "ok",
  "uptime_seconds": 3601.2
}
```

`status` is `"ok"` only when both timescaledb and redis are reachable. Otherwise `"degraded"`.

#### `GET /metrics` — Prometheus metrics

No authentication required. Returns all 15 Prometheus metrics in the standard exposition format. Prometheus scrapes this endpoint automatically.

#### `GET /docs` — Swagger UI

Interactive API documentation with try-it-out support.

---

### Internal Webhook Receivers

These endpoints are used by the default alert configuration for local testing. Replace with your actual retraining pipeline in production.

#### `POST /internal/alert-received`

Logs received alert payloads. Default target for the `local_logger` webhook rule.

#### `POST /internal/retrain-triggered`

Logs retraining triggers. Default target for the `retraining_trigger` webhook rule. In production, replace the URL in `alert_rules.yaml` with your actual pipeline endpoint (GitHub Actions, Airflow, SageMaker Pipelines, etc.).

---

## 10. Alert Rules

Alert rules are defined in `config/alert_rules.yaml` and are evaluated after every drift computation run. No restart is needed — the file is read on each evaluation.

### Rule Schema

```yaml
rules:
  - name: "rule_name"           # Unique identifier for this rule
    method: "ks_test"           # Which drift method to evaluate
    threshold: 0.05             # Numeric threshold value
    operator: "lt"              # Comparison: lt, gt, lte, gte
    severity: "critical"        # ok | info | warning | critical
    window_hours: 24            # Time window for evaluation
    description: "Human readable description"
    retraining_trigger: true    # Whether to fire the retraining webhook
```

**Operators:** `lt` (less than), `gt` (greater than), `lte` (less than or equal), `gte` (greater than or equal)

### Default Rules

```yaml
rules:
  # KS Test — p-value below threshold means drift
  - name: "ks_test_critical"
    method: "ks_test"
    threshold: 0.05
    operator: "lt"          # p-value < 0.05 → critical
    severity: "critical"
    retraining_trigger: true

  - name: "ks_test_warning"
    method: "ks_test"
    threshold: 0.1
    operator: "lt"          # p-value < 0.10 → warning
    severity: "warning"
    retraining_trigger: false

  # PSI — higher score means more drift
  - name: "psi_critical"
    method: "psi"
    threshold: 0.25
    operator: "gt"          # PSI > 0.25 → critical
    severity: "critical"
    retraining_trigger: true

  - name: "psi_warning"
    method: "psi"
    threshold: 0.1
    operator: "gt"          # PSI > 0.10 → warning
    severity: "warning"
    retraining_trigger: false

  # JS Divergence
  - name: "js_divergence_warning"
    method: "js_divergence"
    threshold: 0.1
    operator: "gt"
    severity: "warning"
    retraining_trigger: false

  # SHAP drift — lower rank correlation means more drift
  - name: "shap_drift_critical"
    method: "shap_drift"
    threshold: 0.7
    operator: "lt"          # rank_corr < 0.70 → critical
    severity: "critical"
    retraining_trigger: true
```


### Webhook Configuration

Webhooks are defined in the same file under `webhooks:`. Each webhook can be scoped to specific severity levels.

```yaml
webhooks:
  # Logs every alert locally (default dev configuration)
  - name: "local_logger"
    url: "http://localhost:8001/internal/alert-received"
    enabled: true

  # Fires only on critical severity — triggers retraining pipeline
  - name: "retraining_trigger"
    url: "http://localhost:8001/internal/retrain-triggered"
    enabled: true
    trigger_on_severity: ["critical"]
```

**Webhook payload** (sent on every alert):

```json
{
  "rule_name": "ks_test_critical",
  "model_id": "fraud_detector_v1",
  "feature_name": "amount",
  "method": "ks_test",
  "score": 0.003,
  "threshold": 0.05,
  "severity": "critical",
  "description": "KS test p-value below 0.05 — significant distribution shift",
  "fired_at": "2024-01-15T10:05:00"
}
```

**Retry behavior:** Webhooks are retried up to 3 times with a 2-second wait between attempts (configurable in `config.yaml` under `alerts`). Failures are logged but do not affect drift computation.

### Wiring to a Real Retraining Pipeline

Replace the `url` in `retraining_trigger` with your pipeline endpoint:

```yaml
# GitHub Actions
url: "https://api.github.com/repos/your-org/your-repo/dispatches"

# Airflow (via REST API)
url: "http://airflow:8080/api/v1/dags/retrain_fraud_model/dagRuns"

# AWS SageMaker Pipelines (via API Gateway)
url: "https://your-api-gateway.amazonaws.com/prod/trigger-pipeline"
```

For authenticated webhooks, add auth headers by extending `webhook.py`.

---

## 11. Prometheus Metrics

Argus exposes 15 Prometheus metrics at `GET /metrics`. All metrics use the `argus_` prefix.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `argus_ingest_total` | Counter | `model_id`, `status` | Total feature vectors ingested. `status` is `accepted` or `rejected`. |
| `argus_ingest_batch_size` | Histogram | — | Distribution of batch sizes (buckets: 1, 5, 10, 50, 100, 500, 1000) |
| `argus_drift_score` | Gauge | `model_id`, `feature_name`, `method` | Current drift score for each model/feature/method combination |
| `argus_drift_severity` | Gauge | `model_id`, `feature_name` | Worst severity across all methods: 0=ok, 1=info, 2=warning, 3=critical |
| `argus_drift_runs_total` | Counter | `model_id`, `status` | Total drift computation runs. `status` is `ok`, `skipped`, or `error`. |
| `argus_drift_duration_seconds` | Histogram | `model_id` | Time to compute all drift methods for one model (buckets: 0.1s to 60s) |
| `argus_alerts_fired_total` | Counter | `model_id`, `severity`, `method` | Total alerts fired by the alert engine |
| `argus_webhooks_sent_total` | Counter | `status` | Total webhook calls. `status` is `success` or `failed`. |
| `argus_retraining_triggers_total` | Counter | `model_id` | Total retraining webhooks fired |
| `argus_stream_consumer_lag` | Gauge | — | Number of pending (unprocessed) messages in the Redis Stream |
| `argus_reference_samples` | Gauge | `model_id`, `feature_name` | Number of samples in the reference distribution per feature |
| `argus_production_samples_window` | Gauge | `model_id` | Number of production samples in the current drift window |
| `argus_shap_rank_correlation` | Gauge | `model_id` | Current SHAP feature importance rank correlation vs reference |
| `argus_uptime_seconds` | Gauge | — | Argus server uptime in seconds |

### Example Prometheus Queries

```promql
# Is drift worsening? Rate of critical alerts in the last hour
rate(argus_alerts_fired_total{severity="critical"}[1h])

# Current worst drift severity across all models
max by (model_id) (argus_drift_severity)

# Models with critical drift right now
argus_drift_severity == 3

# Redis consumer lag (should stay near 0)
argus_stream_consumer_lag

# Ingestion rate per model
rate(argus_ingest_total{status="accepted"}[5m])

# Drift computation duration (99th percentile)
histogram_quantile(0.99, rate(argus_drift_duration_seconds_bucket[10m]))

# SHAP drift for all models (low = feature importance shifted)
argus_shap_rank_correlation
```

---

## 12. Grafana Dashboard

The Argus Grafana dashboard is automatically provisioned when the stack starts. No manual setup required.

**Access:** http://localhost:3000 — login with `admin` / `GF_SECURITY_ADMIN_PASSWORD` from your `.env`

**Dashboard panels:**

- **Drift Severity Heatmap** — Live severity (0–3) for every model × feature combination
- **Drift Score Time-Series** — PSI, KS, and JSD scores over time per feature
- **SHAP Rank Correlation** — Model-level feature importance stability trend
- **Alerts Fired** — Alert rate by severity and method
- **Ingest Rate** — Requests per second accepted/rejected per model
- **Redis Stream Lag** — Consumer lag (should be near 0; spikes indicate backpressure)
- **Drift Computation Duration** — P50/P95/P99 latency of drift runs
- **Uptime** — Argus server uptime

The dashboard JSON is at `dashboards/argus.json` and can be imported into any Grafana instance manually if you're not using Docker Compose.


---

## 13. Configuration Reference

### `config/config.yaml`

All non-secret application configuration. Secrets are injected via environment variables.

```yaml
server:
  host: "0.0.0.0"
  port: 8001

timescaledb:
  host: "timescaledb"      # Docker service name (use localhost for local dev)
  port: 5432
  min_pool: 2              # Minimum DB connection pool size
  max_pool: 10             # Maximum DB connection pool size
  # Credentials from: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

redis:
  url: "redis://redis:6379"  # Docker service name
  stream_key: "argus:ingest:stream"
  consumer_group: "argus:consumers"
  consumer_name: "worker-1"
  batch_size: 100            # Messages to read per poll
  block_ms: 1000             # How long to block waiting for messages (ms)
  # Password from: REDIS_PASSWORD

drift:
  schedule_interval_seconds: 300   # Run drift every 5 minutes
  reference_window_days: 30        # How far back to look for reference data on refresh
  production_window_hours: 24      # Production window compared against reference
  min_samples: 10                  # Skip drift if fewer than this many production samples
  methods:                         # Which drift methods to run (remove to disable)
    - "ks_test"
    - "psi"
    - "chi_squared"
    - "js_divergence"
    - "shap_drift"

alerts:
  rules_file: "config/alert_rules.yaml"
  webhook_timeout_seconds: 10
  webhook_max_retries: 3
  webhook_retry_wait_seconds: 2.0

duckdb:
  db_path: ":memory:"    # Use a file path like "/tmp/argus.duckdb" for persistence
  history_days: 30
```

### Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `ARGUS_API_KEY` | Yes (prod) | API key for X-API-Key authentication |
| `POSTGRES_USER` | Yes | TimescaleDB username |
| `POSTGRES_PASSWORD` | Yes | TimescaleDB password |
| `POSTGRES_DB` | Yes | TimescaleDB database name |
| `REDIS_PASSWORD` | Recommended | Redis AUTH password |
| `GF_SECURITY_ADMIN_USER` | No | Grafana admin username (default: admin) |
| `GF_SECURITY_ADMIN_PASSWORD` | Yes | Grafana admin password |
| `ARGUS_WORKERS` | No | Uvicorn worker count (default: 2) |
| `LOG_LEVEL` | No | Log level: debug/info/warning/error (default: info) |

### TimescaleDB Schema

Argus initializes the following tables in TimescaleDB on first start (`scripts/init_timescaledb.sql`):

| Table | Type | Description |
|-------|------|-------------|
| `models` | Regular | Registered model metadata and feature schemas |
| `feature_logs` | Hypertable | Raw feature vectors and predictions (partitioned by `logged_at`) |
| `drift_scores` | Hypertable | Time-series drift scores for all methods (partitioned by `scored_at`) |
| `reference_distributions` | Regular | Stored reference distribution statistics per feature |
| `alert_history` | Hypertable | Record of every fired alert and webhook status |

TimescaleDB hypertables provide automatic time-based partitioning for `feature_logs` and `drift_scores`, enabling fast range queries without manual partition management.

---

## 14. Port Reference

| Service | Host Port | Container Port | Protocol |
|---------|-----------|----------------|----------|
| Argus API | 8001 | 8001 | HTTP |
| TimescaleDB | 5432 | 5432 | PostgreSQL |
| Redis | **6380** | 6379 | Redis |
| Prometheus | 9090 | 9090 | HTTP |
| Grafana | 3000 | 3000 | HTTP |

**Note:** Redis is exposed on host port **6380** (not the default 6379) to avoid conflicts with any existing Redis instance on your machine. Inside Docker networking, services communicate with each other on port 6379 as normal.

If port 6380 conflicts with another service on your machine, change the mapping in `docker-compose.yml`:

```yaml
redis:
  ports:
    - "6381:6379"   # Use 6381 instead
```

---

## 15. Running Tests

The test suite has 40 pytest tests. All external dependencies (Redis, TimescaleDB) are mocked — no Docker container is needed to run tests.

```bash
cd argus/

# Install test dependencies (if running outside Docker)
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_drift_engine.py -v

# Run with coverage
pytest tests/ -v --cov=argus_core --cov-report=term-missing
```

**Test modules:**

| File | What it tests |
|------|--------------|
| `test_ks_test.py` | KS statistic computation, severity mapping, edge cases |
| `test_psi.py` | PSI for numeric and categorical, bin edge cases |
| `test_chi_squared.py` | Chi-squared statistic, missing categories |
| `test_js_divergence.py` | JSD computation, identical distributions, extreme drift |
| `test_shap_drift.py` | Rank correlation, feature mismatch handling, proxy estimation |
| `test_drift_engine.py` | Full drift run orchestration, insufficient samples, scheduling |
| `test_ingest.py` | Ingest service, Redis stream writing, batch ingestion |
| `test_alert_engine.py` | Rule evaluation, webhook firing, severity mapping |
| `conftest.py` | Shared fixtures: mock Redis, mock DB pool, sample feature data |


---

## 16. Demo Scenarios

### Real Model Degradation Demo

`demo/real_degradation_demo.py` is the flagship demo. It simulates a real-world scenario end-to-end:

**Scenario:** A fraud detection model trained on normal transaction patterns. At day 10, fraudsters adapt — transactions shift to higher amounts, late-night hours, and new merchant categories (`crypto_exchange`, `wire_transfer`) that the model has never seen.

**What happens:**
- Days 1–10: Normal traffic. Model accuracy ~88%. Argus severity: `ok`.
- Days 10–15: 30% of traffic is drifted. Feature distributions start shifting. Argus detects drift at the feature level.
- Days 15–30: 100% drifted traffic. Model accuracy collapses to ~60%.

**Argus catches drift ~6 days before accuracy collapse is measurable**, giving an operations team lead time to retrain.

**Running the demo:**

```bash
# Start the stack first
docker compose up --build

# In a separate terminal
cd argus/
python demo/real_degradation_demo.py
```

The demo outputs a day-by-day table to the terminal:

```
  Day    Phase          Acc     AUC   Fraud%  Argus Severity   Top Signal
  ─────  ────────────  ───────  ───── ────────  ──────────────── ──────────────────
  3      normal         0.882  0.941    8.0%   ok               —
  6      normal         0.879  0.938    7.8%   ok               —
  9      normal         0.881  0.940    8.1%   ok               —
  12     early drift    0.851  0.901   12.4%   warning  ←       amount.ks_test=0.042
  15     FULL DRIFT     0.771  0.822   24.1%   critical ←       amount.psi=0.31
  18     FULL DRIFT     0.682  0.741   33.2%   critical         _model.shap_drift=0.58
```

After the run, view live results at:
- http://localhost:3000 (Grafana)
- http://localhost:8001/drift/fraud_detector_v1/latest
- http://localhost:9090

### Synthetic Drift Demo

`demo/synthetic_drift_demo.py` provides a simpler scenario with controllable Gaussian distributions, useful for understanding how each drift method responds to a gradual mean shift.

---

## 17. Production Hardening

The following changes are recommended before running Argus in a production environment.

### Security

**Rotate all default credentials.** The `.env.example` values are placeholders — treat any deployment that uses them as compromised.

```bash
# Generate a strong API key
openssl rand -hex 32

# Generate strong passwords
openssl rand -base64 32
```

**Use HTTPS.** Argus itself speaks plain HTTP. Place it behind a TLS-terminating reverse proxy (nginx, Caddy, AWS ALB) in production. The `X-API-Key` header is transmitted in plaintext over HTTP.

**Restrict network access.** The TimescaleDB port (5432) and Redis port (6380) should not be reachable from outside your internal network. Only the Argus API (8001) and Grafana (3000) need external exposure, and both should be behind the reverse proxy.

**Use Docker secrets or a secrets manager.** For Kubernetes or ECS deployments, inject credentials via Kubernetes Secrets, AWS Secrets Manager, or HashiCorp Vault rather than plaintext environment variables.

### Scale

**Increase worker count.** Set `ARGUS_WORKERS` to the number of CPU cores available:

```env
ARGUS_WORKERS=4
```

**Increase DB connection pool.** Under high ingestion load, increase `max_pool` in `config.yaml`:

```yaml
timescaledb:
  min_pool: 4
  max_pool: 20
```

**Redis memory.** The default Redis configuration caps memory at 512MB with LRU eviction. For high-volume deployments, increase this in `docker-compose.yml`:

```yaml
command: >
  redis-server
  --requirepass ${REDIS_PASSWORD}
  --maxmemory 2gb
  --maxmemory-policy allkeys-lru
```

**Tune drift schedule.** The default drift interval is 5 minutes. For low-volume models you can increase it; for high-stakes real-time models you can decrease it:

```yaml
drift:
  schedule_interval_seconds: 60    # Run every minute for critical models
```

### Data Retention

**TimescaleDB retention policy.** Add a retention policy to automatically drop old feature logs:

```sql
-- Keep feature logs for 90 days, drift scores indefinitely
SELECT add_retention_policy('feature_logs', INTERVAL '90 days');
```

**Prometheus retention.** The default Prometheus retention is 30 days, controlled by the `--storage.tsdb.retention.time=30d` flag in `docker-compose.yml`.

### Observability of Argus Itself

- Alert on `argus_stream_consumer_lag > 1000` — indicates the consumer cannot keep up with ingestion rate
- Alert on `argus_drift_runs_total{status="error"}` — indicates drift computation failures
- Alert on `argus_webhooks_sent_total{status="failed"}` — indicates webhook delivery problems
- Monitor `argus_uptime_seconds` as a dead-man's switch

### Retraining Pipeline Integration

The `POST /internal/retrain-triggered` endpoint is a placeholder. In production, replace the webhook URL in `alert_rules.yaml` with your actual pipeline trigger:

```yaml
webhooks:
  - name: "retraining_trigger"
    url: "https://your-pipeline-endpoint/trigger"
    enabled: true
    trigger_on_severity: ["critical"]
```

The webhook payload contains the model ID, feature name, method, score, and severity — enough for a downstream pipeline to know which model to retrain and why.

### High Availability

For a highly available Argus deployment:

1. Run multiple Argus API instances behind a load balancer. All state is in TimescaleDB and Redis.
2. Use TimescaleDB with streaming replication (or Timescale Cloud with automatic HA).
3. Use Redis Sentinel or Redis Cluster for Redis HA.
4. The APScheduler drift runs will execute on every Argus instance — add distributed locking (Redis-based) if this causes duplicate writes to matter.

---

*Argus version 0.1.0 — built with FastAPI, TimescaleDB, Redis Streams, APScheduler, and scipy.*
