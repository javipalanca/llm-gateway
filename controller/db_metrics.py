# controller/db_metrics.py
#
# PostgreSQL persistence for Prometheus metrics
#
# Almacena snapshots de métricas cada 5 minutos para preservar
# valores de counters entre reinicios del controller
#
# Tabla: metrics_snapshots
#   - timestamp: cuando se tomó el snapshot
#   - model: nombre del modelo
#   - metric_name: nombre de la métrica
#   - metric_type: counter, gauge, histogram
#   - metric_value: valor numérico
#   - labels_json: labels adicionales en JSON
#
# Funcionalidad:
#   - restore_metrics(): Al startup, restaura counters desde DB
#   - persist_metrics(): Cada 5 min, guarda snapshot en DB
#   - cleanup_metrics(): Limpia snapshots antiguos (>30 días)

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import asyncpg

logger = logging.getLogger(__name__)

# Database connection
_db_pool: Optional[asyncpg.Pool] = None

# Database configuration from environment
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://gateway:YOUR_PASSWORD@postgres:5432/gateway"
)

METRICS_RETENTION_DAYS = int(os.environ.get("METRICS_RETENTION_DAYS", "30"))


async def init_db() -> None:
    """Initialize database pool and create tables"""
    global _db_pool
    
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"[DB] Attempt {attempt+1}/{max_retries} to connect to database...", flush=True)
            _db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            print("[DB] ✓ Database connection pool created", flush=True)
            
            # Create tables if not exist
            await _create_tables()
            print("[DB] ✓ Database initialized for metrics persistence", flush=True)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[DB] ⚠ Attempt {attempt+1}/{max_retries} failed, retrying in {retry_delay}s: {e}", flush=True)
                await asyncio.sleep(retry_delay)
            else:
                print(f"[DB] ✗ Failed to initialize database after {max_retries} attempts: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise


async def close_db() -> None:
    """Close database pool"""
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None


async def _create_tables() -> None:
    """Create metrics_snapshots table if not exists"""
    if not _db_pool:
        print("[DB] ERROR: _db_pool is None in _create_tables", flush=True)
        return
    
    print("[DB] Creating tables...", flush=True)
    
    # Ejecutar cada statement por separado
    statements = [
        """CREATE TABLE IF NOT EXISTS metrics_snapshots (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            model VARCHAR(255) NOT NULL,
            metric_name VARCHAR(255) NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            metric_value DOUBLE PRECISION NOT NULL,
            labels_json JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        
        """CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_model_metric 
            ON metrics_snapshots(model, metric_name)""",
        
        """CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_timestamp 
            ON metrics_snapshots(timestamp)""",
        
        """CREATE TABLE IF NOT EXISTS metrics_state (
            id SERIAL PRIMARY KEY,
            model VARCHAR(255) NOT NULL UNIQUE,
            requests_total BIGINT DEFAULT 0,
            model_evictions_total BIGINT DEFAULT 0,
            readiness_probe_failures_total BIGINT DEFAULT 0,
            last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        
        """CREATE INDEX IF NOT EXISTS idx_metrics_state_model ON metrics_state(model)"""
    ]
    
    try:
        async with _db_pool.acquire() as conn:
            for i, statement in enumerate(statements, 1):
                try:
                    print(f"[DB] Executing statement {i}/{len(statements)}...", flush=True)
                    await conn.execute(statement)
                    print(f"[DB] ✓ Statement {i} executed", flush=True)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"[DB] Warning on statement {i}: {e}", flush=True)
        print("[DB] ✓ Metrics tables created/verified", flush=True)
    except Exception as e:
        print(f"[DB] ✗ Failed to create metrics tables: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


async def restore_metrics() -> Dict[str, int]:
    """
    Restore metric counters from database at startup.
    Returns dict with restored values: {model: {metric_name: value}}
    """
    if not _db_pool:
        logger.warning("Database not initialized, skipping metrics restoration")
        return {}
    
    try:
        async with _db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT model, requests_total, model_evictions_total, 
                       readiness_probe_failures_total
                FROM metrics_state
                """
            )
        
        restored = {}
        for row in rows:
            model = row['model']
            restored[model] = {
                'requests_total': row['requests_total'] or 0,
                'model_evictions_total': row['model_evictions_total'] or 0,
                'readiness_probe_failures_total': row['readiness_probe_failures_total'] or 0,
            }
        
        if restored:
            logger.info(f"✓ Restored metrics for {len(restored)} models from database")
            for model, metrics in restored.items():
                logger.debug(f"  {model}: {metrics}")
        
        return restored
    except Exception as e:
        logger.error(f"✗ Failed to restore metrics: {e}")
        return {}


async def persist_metrics(metrics_data: Dict[str, Any]) -> None:
    """
    Persist current metrics state to database.
    
    metrics_data format:
    {
        "model_name": {
            "requests_total": 100,
            "model_evictions_total": 5,
            "readiness_probe_failures_total": 0,
            ...
        }
    }
    """
    if not _db_pool:
        logger.warning("Database not initialized, skipping metrics persistence")
        return
    
    try:
        async with _db_pool.acquire() as conn:
            async with conn.transaction():
                for model, metrics in metrics_data.items():
                    # Insert or update metrics state
                    await conn.execute(
                        """
                        INSERT INTO metrics_state 
                        (model, requests_total, model_evictions_total, 
                         readiness_probe_failures_total, last_updated)
                        VALUES ($1, $2, $3, $4, NOW())
                        ON CONFLICT (model) DO UPDATE SET
                            requests_total = $2,
                            model_evictions_total = $3,
                            readiness_probe_failures_total = $4,
                            last_updated = NOW()
                        """,
                        model,
                        int(metrics.get('requests_total', 0)),
                        int(metrics.get('model_evictions_total', 0)),
                        int(metrics.get('readiness_probe_failures_total', 0))
                    )
                    
                    # Insert snapshot records
                    for metric_name, value in metrics.items():
                        if metric_name in ['requests_total', 'model_evictions_total', 
                                          'readiness_probe_failures_total']:
                            await conn.execute(
                                """
                                INSERT INTO metrics_snapshots
                                (model, metric_name, metric_type, metric_value)
                                VALUES ($1, $2, 'counter', $3)
                                """,
                                model,
                                metric_name,
                                float(value)
                            )
        
        logger.debug(f"✓ Persisted metrics for {len(metrics_data)} models")
    except Exception as e:
        logger.error(f"✗ Failed to persist metrics: {e}")


async def cleanup_old_metrics() -> None:
    """Remove snapshots older than METRICS_RETENTION_DAYS"""
    if not _db_pool:
        return
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=METRICS_RETENTION_DAYS)
        
        async with _db_pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM metrics_snapshots 
                WHERE created_at < $1
                """,
                cutoff_date
            )
        
        # Parse result string: "DELETE <count>"
        deleted_count = int(result.split()[-1]) if result else 0
        if deleted_count > 0:
            logger.info(f"✓ Cleaned up {deleted_count} old metric snapshots")
    except Exception as e:
        logger.error(f"✗ Failed to cleanup old metrics: {e}")


async def get_metric_history(
    model: str,
    metric_name: str,
    hours: int = 24
) -> list:
    """
    Get historical metric values for a model over last N hours.
    Useful for graphs and analysis.
    """
    if not _db_pool:
        return []
    
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        async with _db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT timestamp, metric_value 
                FROM metrics_snapshots
                WHERE model = $1 AND metric_name = $2 AND timestamp > $3
                ORDER BY timestamp ASC
                """,
                model,
                metric_name,
                cutoff_time
            )
        
        return [
            {
                'timestamp': row['timestamp'].isoformat(),
                'value': row['metric_value']
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"✗ Failed to get metric history: {e}")
        return []


async def get_current_metrics_state() -> Dict[str, Dict[str, int]]:
    """Get current persisted metrics state from database"""
    if not _db_pool:
        return {}
    
    try:
        async with _db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT model, requests_total, model_evictions_total,
                       readiness_probe_failures_total, last_updated
                FROM metrics_state
                ORDER BY model
                """
            )
        
        return {
            row['model']: {
                'requests_total': row['requests_total'] or 0,
                'model_evictions_total': row['model_evictions_total'] or 0,
                'readiness_probe_failures_total': row['readiness_probe_failures_total'] or 0,
                'last_updated': row['last_updated'].isoformat() if row['last_updated'] else None,
            }
            for row in rows
        }
    except Exception as e:
        logger.error(f"✗ Failed to get metrics state: {e}")
        return {}
