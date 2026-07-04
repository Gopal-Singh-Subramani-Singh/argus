from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import structlog

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """
    In-memory model registry with optional TimescaleDB persistence.
    Stores model metadata, feature schemas, and reference distributions.
    """

    def __init__(self, db_pool=None):
        self._db = db_pool
        self._models: Dict[str, dict] = {}
        self._schemas: Dict[str, list] = {}
        self._reference: Dict[str, dict] = {}

    async def register(
        self,
        model_id: str,
        name: str,
        version: str,
        features: List[dict],
        metadata: dict = {},
    ) -> dict:
        entry = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "features": features,
            "metadata": metadata,
            "registered_at": datetime.utcnow().isoformat(),
        }
        self._models[model_id] = entry
        self._schemas[model_id] = features

        if self._db:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO models (model_id, name, version, feature_schema)
                    VALUES ($1, $2, $3, $4::jsonb)
                    ON CONFLICT (model_id) DO UPDATE
                    SET name=$2, version=$3, feature_schema=$4::jsonb,
                        updated_at=NOW()
                    """,
                    model_id, name, version, json.dumps(features),
                )
        logger.info(
            "registry.model_registered",
            model_id=model_id,
            features=len(features),
        )
        return entry

    async def set_reference(
        self, model_id: str, feature_name: str,
        feature_type: str, samples: list
    ):
        key = f"{model_id}::{feature_name}"
        self._reference[key] = {
            "type": feature_type,
            "samples": samples[:5000],
            "count": len(samples),
        }
        if self._db:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO reference_distributions
                    (model_id, feature_name, feature_type, stats, sample_count)
                    VALUES ($1, $2, $3, $4::jsonb, $5)
                    ON CONFLICT (model_id, feature_name) DO UPDATE
                    SET stats=$4::jsonb, sample_count=$5,
                        computed_at=NOW()
                    """,
                    model_id, feature_name, feature_type,
                    json.dumps({"samples": samples[:5000]}), len(samples),
                )
        logger.info(
            "registry.reference_set",
            model_id=model_id,
            feature=feature_name,
            n=len(samples),
        )

    async def get_model(self, model_id: str) -> Optional[dict]:
        if model_id in self._models:
            return self._models[model_id]
        if self._db:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM models WHERE model_id=$1", model_id
                )
            if row:
                entry = dict(row)
                self._models[model_id] = entry
                return entry
        return None

    async def get_feature_schema(self, model_id: str) -> List[dict]:
        return self._schemas.get(model_id, [])

    async def list_models(self) -> List[dict]:
        return list(self._models.values())

    def model_exists(self, model_id: str) -> bool:
        return model_id in self._models

    async def delete_model(self, model_id: str):
        self._models.pop(model_id, None)
        self._schemas.pop(model_id, None)
        # Remove reference keys
        keys_to_delete = [k for k in self._reference if k.startswith(f"{model_id}::")]
        for k in keys_to_delete:
            del self._reference[k]
        if self._db:
            async with self._db.acquire() as conn:
                await conn.execute("DELETE FROM models WHERE model_id=$1", model_id)
                await conn.execute(
                    "DELETE FROM reference_distributions WHERE model_id=$1", model_id
                )
        logger.info("registry.model_deleted", model_id=model_id)
