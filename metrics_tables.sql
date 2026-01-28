-- metrics_tables.sql
-- 
-- Script SQL para crear tablas de persistencia de métricas
-- Ejecutar en PostgreSQL después de levantar el servicio:
--
-- psql -U litellm -d litellm < metrics_tables.sql
--

-- Tabla de histórico completo de snapshots
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Índices para queries rápidas
CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_model_metric 
    ON metrics_snapshots(model, metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_timestamp 
    ON metrics_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_created_at 
    ON metrics_snapshots(created_at DESC);

-- Tabla de estado actual (más rápida para restauración)
CREATE TABLE IF NOT EXISTS metrics_state (
    id SERIAL PRIMARY KEY,
    model VARCHAR(255) NOT NULL UNIQUE,
    requests_total BIGINT DEFAULT 0,
    model_evictions_total BIGINT DEFAULT 0,
    readiness_probe_failures_total BIGINT DEFAULT 0,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Índice para lookups por modelo
CREATE INDEX IF NOT EXISTS idx_metrics_state_model 
    ON metrics_state(model);

-- Comentarios para documentación
COMMENT ON TABLE metrics_snapshots IS 'Histórico completo de snapshots de métricas (retención 30 días)';
COMMENT ON TABLE metrics_state IS 'Estado actual de métricas para restauración rápida al startup';
COMMENT ON COLUMN metrics_snapshots.metric_type IS 'Tipo de métrica: counter, gauge, histogram';
COMMENT ON COLUMN metrics_state.requests_total IS 'Total acumulado de requests';
COMMENT ON COLUMN metrics_state.model_evictions_total IS 'Total acumulado de evictions (LRU + TTL)';
COMMENT ON COLUMN metrics_state.readiness_probe_failures_total IS 'Total acumulado de fallos de readiness';

-- Función para limpiar snapshots antiguos
CREATE OR REPLACE FUNCTION cleanup_old_metrics(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM metrics_snapshots
    WHERE created_at < NOW() - MAKE_INTERVAL(days => retention_days);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Verificar creación
SELECT 'Tablas creadas exitosamente' as status;
