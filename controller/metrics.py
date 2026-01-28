# controller/metrics.py
#
# Prometheus metrics definitions for LLM Gateway
#
# Metrics exposed:
# - llm_gateway_requests_total: Total number of requests by model, endpoint, status
# - llm_gateway_request_duration_seconds: Request latency histogram
# - llm_gateway_model_evictions_total: Total evictions by model
# - llm_gateway_vram_free_gib: Free VRAM per GPU
# - llm_gateway_model_startup_time_seconds: Model startup time histogram
# - llm_gateway_in_flight_requests: Current in-flight requests by model
# - llm_gateway_container_status: Container status (1=running, 0=stopped)
# - llm_gateway_model_last_used_timestamp: Last used timestamp per model
# - llm_gateway_vram_check_duration_seconds: VRAM check duration
# - llm_gateway_readiness_probe_duration_seconds: Readiness probe duration
#
# Note: Counters can be incremented from restored values via restore_counter()
# to support persistence across restarts.

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ============================================
# Request Metrics
# ============================================

requests_total = Counter(
    'llm_gateway_requests_total',
    'Total number of requests',
    ['model', 'endpoint', 'status']
)

request_duration_seconds = Histogram(
    'llm_gateway_request_duration_seconds',
    'Request duration in seconds',
    ['model', 'endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf'))
)

# ============================================
# Model Lifecycle Metrics
# ============================================

model_evictions_total = Counter(
    'llm_gateway_model_evictions_total',
    'Total number of model evictions',
    ['model', 'reason']
)

model_startup_time_seconds = Histogram(
    'llm_gateway_model_startup_time_seconds',
    'Model startup time in seconds',
    ['model'],
    buckets=(5.0, 10.0, 30.0, 60.0, 90.0, 120.0, 180.0, 300.0, float('inf'))
)

in_flight_requests = Gauge(
    'llm_gateway_in_flight_requests',
    'Current number of in-flight requests',
    ['model']
)

model_last_used_timestamp = Gauge(
    'llm_gateway_model_last_used_timestamp',
    'Unix timestamp when model was last used',
    ['model']
)

# ============================================
# Container Status Metrics
# ============================================

container_status = Gauge(
    'llm_gateway_container_status',
    'Container status (1=running, 0=stopped, -1=error)',
    ['model', 'container_name']
)

# ============================================
# VRAM Metrics
# ============================================

vram_free_gib = Gauge(
    'llm_gateway_vram_free_gib',
    'Free VRAM in GiB per GPU',
    ['gpu_id']
)

vram_total_gib = Gauge(
    'llm_gateway_vram_total_gib',
    'Total VRAM in GiB per GPU',
    ['gpu_id']
)

vram_check_duration_seconds = Histogram(
    'llm_gateway_vram_check_duration_seconds',
    'Duration of VRAM check operations in seconds',
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, float('inf'))
)

# ============================================
# Model Info Metrics
# ============================================

model_info = Gauge(
    'llm_gateway_model_info',
    'Model information (1=loaded, 0=not loaded) with metadata in labels',
    ['model', 'gpu', 'warm', 'priority']
)

# ============================================
# Readiness Probe Metrics
# ============================================

readiness_probe_duration_seconds = Histogram(
    'llm_gateway_readiness_probe_duration_seconds',
    'Duration of readiness probe operations in seconds',
    ['model'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float('inf'))
)

readiness_probe_failures_total = Counter(
    'llm_gateway_readiness_probe_failures_total',
    'Total number of readiness probe failures',
    ['model']
)

# ============================================
# Helper Functions
# ============================================

def get_metrics_content():
    """Generate Prometheus metrics content"""
    return generate_latest()

def get_metrics_content_type():
    """Get Prometheus metrics content type"""
    return CONTENT_TYPE_LATEST


def restore_counter(counter, labels: dict, value: int) -> None:
    """
    Restore a counter to a specific value.
    Increments the counter by (value - current_value).
    
    Args:
        counter: prometheus_client.Counter object
        labels: dict with label names as keys, label values as values
        value: target value to restore to
    """
    try:
        current = counter.labels(**labels)._value.get()
        increment = value - current
        if increment > 0:
            current.inc(increment)
    except Exception:
        # If counter doesn't have this label combo yet, just inc to value
        counter.labels(**labels).inc(value)


def get_counter_value(counter, labels: dict) -> int:
    """Get current value of a counter with specific labels"""
    try:
        return int(counter.labels(**labels)._value.get())
    except Exception:
        return 0


def collect_counters_state() -> dict:
    """
    Collect current state of all counter metrics.
    Returns dict: {metric_name: {label_combo: value}}
    """
    state = {}
    
    # Collect requests_total
    state['requests_total'] = {}
    for sample in requests_total.collect()[0].samples:
        if sample.name == 'llm_gateway_requests_total':
            labels = sample.labels.get('model', 'unknown')
            state['requests_total'][labels] = state['requests_total'].get(labels, 0) + sample.value
    
    # Collect model_evictions_total
    state['model_evictions_total'] = {}
    for sample in model_evictions_total.collect()[0].samples:
        if sample.name == 'llm_gateway_model_evictions_total':
            labels = sample.labels.get('model', 'unknown')
            state['model_evictions_total'][labels] = state['model_evictions_total'].get(labels, 0) + sample.value
    
    # Collect readiness_probe_failures_total
    state['readiness_probe_failures_total'] = {}
    for sample in readiness_probe_failures_total.collect()[0].samples:
        if sample.name == 'llm_gateway_readiness_probe_failures_total':
            labels = sample.labels.get('model', 'unknown')
            state['readiness_probe_failures_total'][labels] = state['readiness_probe_failures_total'].get(labels, 0) + sample.value
    
    return state
