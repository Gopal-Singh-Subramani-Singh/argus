from __future__ import annotations
import time
from prometheus_client import Counter, Histogram, Gauge

INGEST_TOTAL = Counter(
    "argus_ingest_total",
    "Total feature vectors ingested",
    ["model_id", "status"],
)

INGEST_BATCH_SIZE = Histogram(
    "argus_ingest_batch_size",
    "Size of ingested batches",
    buckets=[1, 5, 10, 50, 100, 500, 1000],
)

DRIFT_SCORE = Gauge(
    "argus_drift_score",
    "Current drift score per model/feature/method",
    ["model_id", "feature_name", "method"],
)

DRIFT_SEVERITY = Gauge(
    "argus_drift_severity",
    "Drift severity level: 0=ok, 1=info, 2=warning, 3=critical",
    ["model_id", "feature_name"],
)

DRIFT_RUNS_TOTAL = Counter(
    "argus_drift_runs_total",
    "Total drift computation runs",
    ["model_id", "status"],
)

DRIFT_DURATION = Histogram(
    "argus_drift_duration_seconds",
    "Time to compute drift for one model",
    ["model_id"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

ALERTS_FIRED = Counter(
    "argus_alerts_fired_total",
    "Total alerts fired",
    ["model_id", "severity", "method"],
)

WEBHOOKS_SENT = Counter(
    "argus_webhooks_sent_total",
    "Total webhook calls made",
    ["status"],
)

RETRAINING_TRIGGERS = Counter(
    "argus_retraining_triggers_total",
    "Total retraining webhook triggers",
    ["model_id"],
)

STREAM_LAG = Gauge(
    "argus_stream_consumer_lag",
    "Redis Streams consumer lag (pending messages)",
)

REFERENCE_SAMPLES = Gauge(
    "argus_reference_samples",
    "Number of samples in reference distribution",
    ["model_id", "feature_name"],
)

PRODUCTION_SAMPLES = Gauge(
    "argus_production_samples_window",
    "Production samples in current drift window",
    ["model_id"],
)

SHAP_RANK_CORRELATION = Gauge(
    "argus_shap_rank_correlation",
    "SHAP feature importance rank correlation vs reference",
    ["model_id"],
)

UPTIME = Gauge("argus_uptime_seconds", "Argus server uptime in seconds")

_START = time.time()


def update_uptime() -> float:
    elapsed = time.time() - _START
    UPTIME.set(elapsed)
    return elapsed


SEVERITY_MAP = {"ok": 0, "info": 1, "warning": 2, "critical": 3}
