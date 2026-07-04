from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
from datetime import datetime
import uuid


class FeatureType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class DriftSeverity(str, Enum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FeatureSchema(BaseModel):
    name: str
    type: FeatureType
    description: Optional[str] = None


class ModelRegistrationRequest(BaseModel):
    model_id: str
    name: str
    version: str
    features: List[FeatureSchema]
    metadata: Dict[str, Any] = {}


class ModelRegistrationResponse(BaseModel):
    model_id: str
    name: str
    version: str
    registered_at: datetime
    status: str = "registered"


class IngestRequest(BaseModel):
    model_id: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    features: Dict[str, Any]
    prediction: Optional[Any] = None
    label: Optional[Any] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class BatchIngestRequest(BaseModel):
    records: List[IngestRequest]


class IngestResponse(BaseModel):
    request_id: str
    status: str = "accepted"
    queued: bool = True


class DriftScore(BaseModel):
    feature_name: str
    method: str
    score: float
    p_value: Optional[float] = None
    severity: DriftSeverity = DriftSeverity.OK
    sample_count: int = 0
    scored_at: datetime = Field(default_factory=datetime.utcnow)


class DriftReport(BaseModel):
    model_id: str
    window_start: datetime
    window_end: datetime
    total_samples: int
    scores: List[DriftScore]
    overall_severity: DriftSeverity
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ReferenceSetRequest(BaseModel):
    model_id: str
    features_data: List[Dict[str, Any]]
    compute_shap: bool = False
    model_artifact: Optional[str] = None


class AlertRule(BaseModel):
    name: str
    method: str
    threshold: float
    operator: Literal["lt", "gt", "lte", "gte"]
    severity: DriftSeverity
    window_hours: int = 24
    description: str = ""
    retraining_trigger: bool = False


class Alert(BaseModel):
    rule_name: str
    model_id: str
    feature_name: Optional[str]
    method: str
    score: float
    threshold: float
    severity: DriftSeverity
    description: str
    fired_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    timescaledb: str
    redis: str
    uptime_seconds: float


class ModelStatusResponse(BaseModel):
    model_id: str
    name: str
    version: str
    total_logs: int
    last_logged: Optional[datetime]
    latest_drift_severity: DriftSeverity
    registered_at: datetime


class HistoryQueryResponse(BaseModel):
    model_id: str
    feature_name: str
    method: str
    history: List[Dict[str, Any]]
    days: int
