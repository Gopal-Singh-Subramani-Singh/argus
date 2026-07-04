from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))
    workers: int = Field(default_factory=lambda: int(os.getenv("ARGUS_WORKERS", "1")))


class TimescaleConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "argus"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "argus"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "argus"))
    min_pool: int = 2
    max_pool: int = 10

    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    def asyncpg_dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379"
    password: Optional[str] = Field(
        default_factory=lambda: os.getenv("REDIS_PASSWORD") or None
    )
    stream_key: str = "argus:ingest:stream"
    consumer_group: str = "argus:consumers"
    consumer_name: str = "worker-1"
    batch_size: int = 100
    block_ms: int = 1000

    def full_url(self) -> str:
        if self.password:
            # inject password into url: redis://:password@host:port
            base = self.url.replace("redis://", f"redis://:{self.password}@")
            return base
        return self.url


class DriftMethodsConfig(BaseModel):
    schedule_interval_seconds: int = 300
    reference_window_days: int = 30
    production_window_hours: int = 24
    min_samples: int = 10
    methods: List[str] = [
        "ks_test", "psi", "chi_squared", "js_divergence", "shap_drift"
    ]


class AlertConfig(BaseModel):
    rules_file: str = "config/alert_rules.yaml"
    webhook_timeout_seconds: int = 10
    webhook_max_retries: int = 3
    webhook_retry_wait_seconds: float = 2.0


class DuckDBConfig(BaseModel):
    db_path: str = ":memory:"
    history_days: int = 30


class SecurityConfig(BaseModel):
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ARGUS_API_KEY")
    )
    # Endpoints exempt from API key checks
    public_paths: List[str] = ["/", "/health", "/metrics", "/docs", "/openapi.json"]


class ArgusConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    timescaledb: TimescaleConfig = Field(default_factory=TimescaleConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    drift: DriftMethodsConfig = Field(default_factory=DriftMethodsConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


_config: Optional[ArgusConfig] = None


def load_config(path: str = "config/config.yaml") -> ArgusConfig:
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        # Remove keys not in ArgusConfig to avoid validation errors
        # (e.g. prometheus key)
        valid_keys = ArgusConfig.model_fields.keys()
        data = {k: v for k, v in data.items() if k in valid_keys}
        return ArgusConfig(**data)
    return ArgusConfig()


def get_config() -> ArgusConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    global _config
    _config = None
