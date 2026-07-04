CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    feature_schema  JSONB NOT NULL,
    reference_stats JSONB,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Raw feature log (hypertable)
CREATE TABLE IF NOT EXISTS feature_logs (
    logged_at    TIMESTAMPTZ NOT NULL,
    model_id     TEXT NOT NULL REFERENCES models(model_id),
    request_id   UUID NOT NULL,
    features     JSONB NOT NULL,
    prediction   JSONB,
    label        JSONB,
    metadata     JSONB
);

SELECT create_hypertable(
    'feature_logs', 'logged_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_feature_logs_model
    ON feature_logs (model_id, logged_at DESC);

-- Drift scores (hypertable)
CREATE TABLE IF NOT EXISTS drift_scores (
    scored_at    TIMESTAMPTZ NOT NULL,
    model_id     TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    method       TEXT NOT NULL,
    score        DOUBLE PRECISION NOT NULL,
    p_value      DOUBLE PRECISION,
    severity     TEXT,
    sample_count INTEGER,
    metadata     JSONB
);

SELECT create_hypertable(
    'drift_scores', 'scored_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_drift_scores_model_feature
    ON drift_scores (model_id, feature_name, scored_at DESC);

CREATE INDEX IF NOT EXISTS idx_drift_scores_method
    ON drift_scores (method, scored_at DESC);

-- Alert history
CREATE TABLE IF NOT EXISTS alert_history (
    alerted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_id      TEXT NOT NULL,
    rule_name     TEXT NOT NULL,
    feature_name  TEXT,
    method        TEXT NOT NULL,
    score         DOUBLE PRECISION NOT NULL,
    threshold     DOUBLE PRECISION NOT NULL,
    severity      TEXT NOT NULL,
    webhook_fired BOOLEAN DEFAULT FALSE,
    retrain_fired BOOLEAN DEFAULT FALSE,
    metadata      JSONB
);

SELECT create_hypertable(
    'alert_history', 'alerted_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Reference distributions (stored as compressed JSON stats)
CREATE TABLE IF NOT EXISTS reference_distributions (
    model_id       TEXT NOT NULL,
    feature_name   TEXT NOT NULL,
    feature_type   TEXT NOT NULL,
    stats          JSONB NOT NULL,
    sample_count   INTEGER NOT NULL,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_id, feature_name)
);
