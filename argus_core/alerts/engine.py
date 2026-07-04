from __future__ import annotations
import yaml
from pathlib import Path
from typing import List
from datetime import datetime
import structlog

from argus_core.models import AlertRule, Alert, DriftScore, DriftSeverity
from argus_core.metrics import ALERTS_FIRED
from argus_core.alerts.webhook import WebhookSender
from config.settings import get_config

logger = structlog.get_logger(__name__)

OPERATORS = {
    "lt": lambda score, thresh: score < thresh,
    "gt": lambda score, thresh: score > thresh,
    "lte": lambda score, thresh: score <= thresh,
    "gte": lambda score, thresh: score >= thresh,
}


class AlertEngine:
    def __init__(self):
        self._rules: List[AlertRule] = []
        self._webhooks: List[dict] = []
        self._webhook_sender: WebhookSender = WebhookSender()
        self._load_rules()

    def _load_rules(self):
        cfg = get_config()
        rules_path = Path(cfg.alerts.rules_file)
        if not rules_path.exists():
            logger.warning("alert_engine.rules_file_not_found", path=str(rules_path))
            return
        with open(rules_path) as f:
            data = yaml.safe_load(f)
        self._rules = [AlertRule(**r) for r in data.get("rules", [])]
        self._webhooks = data.get("webhooks", [])
        logger.info("alert_engine.loaded", rules=len(self._rules))

    async def evaluate(
        self, model_id: str, scores: List[DriftScore]
    ) -> List[Alert]:
        fired: List[Alert] = []

        for rule in self._rules:
            matching = [
                s for s in scores
                if s.method == rule.method
            ]
            for score_obj in matching:
                op = OPERATORS.get(rule.operator)
                if op is None:
                    continue
                if op(score_obj.score, rule.threshold):
                    alert = Alert(
                        rule_name=rule.name,
                        model_id=model_id,
                        feature_name=score_obj.feature_name,
                        method=rule.method,
                        score=score_obj.score,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        description=rule.description,
                    )
                    fired.append(alert)
                    ALERTS_FIRED.labels(
                        model_id=model_id,
                        severity=rule.severity.value,
                        method=rule.method,
                    ).inc()
                    logger.warning(
                        "alert.fired",
                        rule=rule.name,
                        model=model_id,
                        feature=score_obj.feature_name,
                        score=round(score_obj.score, 4),
                        threshold=rule.threshold,
                        severity=rule.severity.value,
                    )
                    await self._dispatch_webhooks(alert, rule)

        return fired

    async def _dispatch_webhooks(self, alert: Alert, rule: AlertRule):
        for wh in self._webhooks:
            if not wh.get("enabled", True):
                continue
            trigger_sevs = wh.get("trigger_on_severity")
            if trigger_sevs and alert.severity.value not in trigger_sevs:
                continue
            await self._webhook_sender.send(wh["url"], alert.model_dump())

        if rule.retraining_trigger:
            retrain_webhooks = [
                w for w in self._webhooks
                if "retrain" in w.get("name", "")
            ]
            for wh in retrain_webhooks:
                if wh.get("enabled", True):
                    await self._webhook_sender.send(
                        wh["url"],
                        {
                            "event": "retraining_triggered",
                            "model_id": alert.model_id,
                            "reason": alert.rule_name,
                            "score": alert.score,
                            "severity": alert.severity.value,
                            "fired_at": alert.fired_at.isoformat(),
                        },
                    )
