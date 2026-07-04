from __future__ import annotations
from typing import Dict, Any
import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from argus_core.metrics import WEBHOOKS_SENT, RETRAINING_TRIGGERS
from config.settings import get_config

logger = structlog.get_logger(__name__)
_tenacity_logger = logging.getLogger("tenacity")


class WebhookSender:
    async def send(self, url: str, payload: Dict[str, Any]) -> bool:
        cfg     = get_config()
        timeout = cfg.alerts.webhook_timeout_seconds
        retries = cfg.alerts.webhook_max_retries

        serialized = {}
        for k, v in payload.items():
            if hasattr(v, "isoformat"):
                serialized[k] = v.isoformat()
            elif hasattr(v, "value"):
                serialized[k] = v.value
            else:
                serialized[k] = v

        async def _attempt():
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=serialized)
                resp.raise_for_status()
            return True

        for attempt in range(1, retries + 1):
            try:
                await _attempt()
                WEBHOOKS_SENT.labels(status="ok").inc()
                logger.info("webhook.sent", url=url, attempt=attempt)
                return True
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < retries:
                    wait = cfg.alerts.webhook_retry_wait_seconds * attempt
                    logger.warning(
                        "webhook.retrying",
                        url=url,
                        attempt=attempt,
                        wait_s=wait,
                        error=str(exc),
                    )
                    import asyncio
                    await asyncio.sleep(wait)
                else:
                    WEBHOOKS_SENT.labels(status="error").inc()
                    logger.warning(
                        "webhook.failed",
                        url=url,
                        attempts=retries,
                        error=str(exc),
                    )
                    return False
            except Exception as exc:
                WEBHOOKS_SENT.labels(status="error").inc()
                logger.warning("webhook.failed", url=url, error=str(exc))
                return False

        return False
