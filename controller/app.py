# controller/app.py
#
# FastAPI "Model Controller" for vLLM containers + OpenAI-compatible proxy to LiteLLM,
# with PRE-CHECK VRAM + PREVENTIVE EVICTION (LRU) before starting a new model.
#
# Key features:
# - OpenAI-compatible endpoints under /v1/*
# - For POST requests with JSON body containing "model":
#     1) PRE-CHECK GPU free VRAM (via a standard CUDA container running nvidia-smi)
#     2) If insufficient VRAM, PREVENTIVE EVICTION of idle models on the same GPU(s) (LRU, not warm, not in_flight)
#     3) Start/create the requested vLLM container
#     4) Readiness probe on /v1/models
#     5) Proxy request to LiteLLM (preserves Authorization)
# - Background reaper stops idle models after TTL
#
# Requirements:
#   pip install fastapi uvicorn[standard] httpx pyyaml docker
#
# Environment variables:
#   LITELLM_BASE_URL            default: http://10.10.1.67:9001
#   MODELS_CONFIG               default: /app/models.yaml
#   IDLE_TTL_MINUTES            default: 20
#   MODEL_READY_TIMEOUT_SEC     default: 300
#   MODEL_READY_POLL_SEC        default: 1
#   VLLM_HOST_IP                default: 10.10.1.67
#
# VRAM pre-check settings:
#   CUDA_SMI_IMAGE              default: nvidia/cuda:12.4.1-base-ubuntu22.04
#   EVICT_RETRY_LIMIT           default: 4      # maximum number of evictions before giving up
#   EVICT_AFTER_FAILED_START    default: 1      # if start/readiness fails, try eviction+retry this many times
#
# models.yaml additions (recommended):
#   models:
#     deepseek-r1:
#       gpu: 1                    # 0, 1, or "all" (if model consumes both)
#       min_free_gib: 60          # required free VRAM on that GPU before starting (include safety margin)
#       warm: false               # if true, never evict/stop
#       priority: 50              # higher means less likely to be evicted (optional)
#       ...
#
# Notes:
# - This controller uses the Docker SDK; no 'docker' CLI required in the container.
# - It needs /var/run/docker.sock mounted and GPUs available to the Docker engine.
# - Pre-check uses a standard CUDA image to execute nvidia-smi, so the controller container
#   does not need nvidia-smi installed.

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import docker
import httpx
import yaml
from docker.errors import APIError, NotFound
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Import Prometheus metrics
import metrics
import db_metrics

# -----------------------------
# Configuration
# -----------------------------

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://10.10.1.67:9001")
MODELS_CONFIG = os.environ.get("MODELS_CONFIG", "/app/models.yaml")
IDLE_TTL_MINUTES = int(os.environ.get("IDLE_TTL_MINUTES", "20"))
MODEL_READY_TIMEOUT_SEC = int(os.environ.get("MODEL_READY_TIMEOUT_SEC", "300"))
MODEL_READY_POLL_SEC = float(os.environ.get("MODEL_READY_POLL_SEC", "1"))
VLLM_HOST_IP = os.environ.get("VLLM_HOST_IP", "10.10.1.67")

CUDA_SMI_IMAGE = os.environ.get("CUDA_SMI_IMAGE", "nvidia/cuda:12.4.1-base-ubuntu22.04")
EVICT_RETRY_LIMIT = int(os.environ.get("EVICT_RETRY_LIMIT", "4"))
EVICT_AFTER_FAILED_START = int(os.environ.get("EVICT_AFTER_FAILED_START", "1"))

http_client = httpx.AsyncClient(timeout=None)

# -------
# Background task functions
# -------

async def _reaper_background():
    """Background task to clean up idle containers"""
    import asyncio
    ttl = IDLE_TTL_MINUTES * 60
    while True:
        try:
            now = _now()
            for model_key, spec in MODELS.items():
                name = spec.get("container_name")
                if not name:
                    continue

                if _is_warm(spec):
                    continue

                lu = last_used.get(model_key)
                if lu is None:
                    continue

                if in_flight.get(model_key, 0) > 0:
                    continue

                if (now - lu) > ttl:
                    if container_running(name):
                        docker_stop(name)
                        metrics.model_evictions_total.labels(model=model_key, reason="ttl_idle").inc()
                        metrics.container_status.labels(model=model_key, container_name=name).set(0)
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[REAPER] Error: {e}", flush=True)
            await asyncio.sleep(30)


async def _persister_background():
    """Background task to persist metrics every 5 minutes"""
    import asyncio
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            
            # Collect all counter states
            state_to_save = {}
            for model_key in MODELS.keys():
                try:
                    state_to_save[model_key] = {
                        'requests_total': metrics.requests_total.labels(
                            model=model_key, endpoint='chat/completions', status='success'
                        )._value.get(),
                        'model_evictions_total': sum(
                            metrics.model_evictions_total.labels(model=model_key, reason=r)._value.get()
                            for r in ['lru_eviction', 'ttl_idle']
                        ),
                        'readiness_probe_failures_total': metrics.readiness_probe_failures_total.labels(
                            model=model_key
                        )._value.get(),
                    }
                except Exception:
                    pass
            
            await db_metrics.persist_metrics(state_to_save)
            await db_metrics.cleanup_old_metrics()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[PERSISTER] Error: {e}", flush=True)


async def _vram_sampler_background():
    """Background task to sample VRAM free/total for all GPUs periodically."""
    import asyncio
    while True:
        try:
            ids = _list_gpu_indices()
            print(f"[VRAM] sample tick, gpu_ids={ids}", flush=True)
            if not ids:
                # Emit zeroed metrics so Grafana shows something even if no GPU detected
                metrics.vram_free_gib.labels(gpu_id="0").set(0)
                metrics.vram_total_gib.labels(gpu_id="0").set(0)
                await asyncio.sleep(30)
                continue
            # Ensure series exist for all discovered GPUs before sampling
            for gid in ids:
                metrics.vram_free_gib.labels(gpu_id=str(gid)).set(0)
                metrics.vram_total_gib.labels(gpu_id=str(gid)).set(0)
            for gid in ids:
                try:
                    free = get_free_vram_gib(gid)
                    gauge_free = metrics.vram_free_gib.labels(gpu_id=str(gid))
                    gauge_total = metrics.vram_total_gib.labels(gpu_id=str(gid))
                    if free is None:
                        # ensure series exists even on failure
                        gauge_free.set(gauge_free._value.get() or 0)
                        gauge_total.set(gauge_total._value.get() or 0)
                    else:
                        total = gauge_total._value.get()
                        print(f"[VRAM] GPU {gid} free={free:.2f} GiB total={total:.2f} GiB", flush=True)
                except Exception as inner:
                    print(f"[VRAM] Error sampling GPU {gid}: {inner}", flush=True)
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[VRAM] Error: {e}", flush=True)
            await asyncio.sleep(30)


async def _model_info_sampler_background():
    """Background task to update model info metrics periodically."""
    import asyncio
    while True:
        try:
            print(f"[MODEL_INFO] sample tick", flush=True)
            for model_key, spec in MODELS.items():
                # Determine if model is running
                container_name = spec.get("container_name")
                is_running = 1 if (container_name and container_running(container_name)) else 0
                
                # Get model characteristics
                warm_val = "true" if _is_warm(spec) else "false"
                priority_val = str(_priority(spec))
                
                # Get GPU assignment from config (always the same, regardless of running state)
                gpu_req = _requested_gpus(spec)
                if gpu_req == "all":
                    gpu_str = "all"
                elif isinstance(gpu_req, list):
                    gpu_str = ",".join(str(g) for g in gpu_req)
                else:
                    gpu_str = str(gpu_req)
                
                # Update metric
                metrics.model_info.labels(
                    model=model_key,
                    gpu=gpu_str,
                    warm=warm_val,
                    priority=priority_val
                ).set(is_running)
                
                if is_running:
                    print(f"[MODEL_INFO] {model_key}: running, gpu={gpu_str}, warm={warm_val}, priority={priority_val}", flush=True)
            
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[MODEL_INFO] Error: {e}", flush=True)
            await asyncio.sleep(30)


# -------
# App Startup/Shutdown
# -------

import contextlib

# Background task references
_reaper_task = None
_persister_task = None
_vram_sampler_task = None
_model_info_sampler_task = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown"""
    import asyncio
    global _reaper_task, _persister_task, _vram_sampler_task, _model_info_sampler_task
    
    # Startup
    print("=[LIFESPAN STARTUP]=", flush=True)
    try:
        print("[DB] Initializing database...", flush=True)
        await db_metrics.init_db()
        print("[DB] ✓ Database initialized", flush=True)
        await db_metrics.restore_metrics()
        print("[DB] ✓ Metrics restored", flush=True)
    except Exception as e:
        print(f"[DB] ✗ Failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # One-shot VRAM sample at startup so gauges are not empty
    try:
        gpu_ids = _list_gpu_indices()
        if not gpu_ids:
            metrics.vram_free_gib.labels(gpu_id="0").set(0)
            metrics.vram_total_gib.labels(gpu_id="0").set(0)
        else:
            for gid in gpu_ids:
                get_free_vram_gib(gid)
        print("[VRAM] Initial sample done", flush=True)
    except Exception as e:
        print(f"[VRAM] Initial sample error: {e}", flush=True)
    
    # Start background tasks
    print("[TASKS] Starting background tasks...", flush=True)
    _reaper_task = asyncio.create_task(_reaper_background())
    _persister_task = asyncio.create_task(_persister_background())
    _vram_sampler_task = asyncio.create_task(_vram_sampler_background())
    _model_info_sampler_task = asyncio.create_task(_model_info_sampler_background())
    print("[TASKS] ✓ Background tasks started", flush=True)
    
    yield
    
    # Shutdown
    print("=[LIFESPAN SHUTDOWN]=", flush=True)
    
    # Cancel background tasks
    if _reaper_task:
        _reaper_task.cancel()
    if _persister_task:
        _persister_task.cancel()
    if _vram_sampler_task:
        _vram_sampler_task.cancel()
    if _model_info_sampler_task:
        _model_info_sampler_task.cancel()
    
    try:
        # Final metrics persistence
        final_state = {}
        for model_key in MODELS.keys():
            try:
                final_state[model_key] = {
                    'requests_total': metrics.requests_total.labels(
                        model=model_key, endpoint='chat/completions', status='success'
                    )._value.get(),
                    'model_evictions_total': sum(
                        metrics.model_evictions_total.labels(model=model_key, reason=r)._value.get()
                        for r in ['lru_eviction', 'ttl_idle']
                    ),
                    'readiness_probe_failures_total': metrics.readiness_probe_failures_total.labels(
                        model=model_key
                    )._value.get(),
                }
            except Exception:
                pass
        
        if final_state:
            await db_metrics.persist_metrics(final_state)
        
        await db_metrics.close_db()
        print("[DB] ✓ Database closed", flush=True)
    except Exception as e:
        print(f"[DB] ✗ Shutdown failed: {e}", flush=True)

app = FastAPI(lifespan=lifespan)

# last_used: last time a given model alias served traffic through controller
last_used: Dict[str, float] = {}
# in_flight: number of in-flight requests per model alias
in_flight: Dict[str, int] = {}

# Docker client (talks to docker daemon via mounted socket)
docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock")


def load_models_config() -> Dict[str, Any]:
    with open(MODELS_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    models = cfg.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models.yaml must contain a top-level key 'models' as a mapping.")
    return models


MODELS: Dict[str, Dict[str, Any]] = load_models_config()

# -----------------------------
# Helpers: Parsing & Common
# -----------------------------


def _now() -> float:
    return time.time()


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _extract_model_from_json(body: bytes) -> Optional[str]:
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    m = data.get("model")
    return m if isinstance(m, str) else None


def _is_stream_requested(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8"))
        return bool(payload.get("stream", False))
    except Exception:
        return False


def _strip_hop_by_hop(headers: Dict[str, str]) -> Dict[str, str]:
    h = {k.lower(): v for k, v in headers.items()}
    for k in [
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    ]:
        h.pop(k, None)
    return h


def _model_spec(model_key: str) -> Dict[str, Any]:
    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Known: {list(MODELS.keys())}")
    return MODELS[model_key]


def _is_warm(spec: Dict[str, Any]) -> bool:
    return bool(spec.get("warm", False))


def _priority(spec: Dict[str, Any]) -> int:
    # Higher priority -> less likely to be evicted
    try:
        return int(spec.get("priority", 0))
    except Exception:
        return 0


def _requested_gpus(spec: Dict[str, Any]) -> Union[str, int, List[int]]:
    """
    Determine which GPU(s) this model conceptually requires.
    Precedence:
      1) spec['gpu'] if present: 0, 1, or "all"
      2) derive from spec['gpus'] ("device=0", "device=1", "all")
    Returns:
      - "all"  OR
      - int gpu_id  OR
      - list[int] for multi-gpu if user provided list (rare)
    """
    if "gpu" in spec:
        g = spec["gpu"]
        if isinstance(g, str) and g.lower() == "all":
            return "all"
        if isinstance(g, int):
            return g
        if isinstance(g, str) and g.isdigit():
            return int(g)
        if isinstance(g, list) and all(isinstance(x, int) for x in g):
            return g
        raise ValueError(f"Invalid gpu field in models.yaml: {g}")

    gpus = str(spec.get("gpus", "all"))
    if gpus == "all":
        return "all"
    if gpus.startswith("device="):
        return int(gpus.split("=", 1)[1])
    if gpus.isdigit():
        return int(gpus)
    # fallback: assume "all"
    return "all"


def _gpu_list_from_requested(req: Union[str, int, List[int]]) -> List[int]:
    if req == "all":
        # We'll discover GPU count dynamically; for precheck we query all available indices.
        # For eviction, "all" means anything running is a candidate.
        return []
    if isinstance(req, int):
        return [req]
    if isinstance(req, list):
        return req
    return []


def _min_free_gib(spec: Dict[str, Any]) -> Optional[float]:
    v = spec.get("min_free_gib", None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        raise ValueError(f"min_free_gib must be numeric, got: {v}")


# -----------------------------
# Helpers: Docker SDK (vLLM containers)
# -----------------------------


def _get_container(name: str):
    return docker_client.containers.get(name)


def container_exists(name: str) -> bool:
    try:
        _get_container(name)
        return True
    except NotFound:
        return False


def container_running(name: str) -> bool:
    try:
        c = _get_container(name)
        c.reload()
        return c.status == "running"
    except NotFound:
        return False


def docker_start(name: str) -> None:
    c = _get_container(name)
    c.start()


def docker_stop(name: str) -> None:
    try:
        c = _get_container(name)
        c.stop(timeout=20)
    except NotFound:
        pass


def docker_remove(name: str) -> None:
    try:
        c = _get_container(name)
        c.remove(force=True)
    except NotFound:
        pass


def _parse_volumes(volumes_list: list[str]) -> Dict[str, Dict[str, str]]:
    volumes: Dict[str, Dict[str, str]] = {}
    for v in volumes_list:
        parts = v.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid volume spec '{v}'. Expected 'host:container[:mode]'.")
        host_path = parts[0]
        container_path = parts[1]
        mode = parts[2] if len(parts) >= 3 else "rw"
        volumes[host_path] = {"bind": container_path, "mode": mode}
    return volumes


def _parse_gpu_request(gpus: str):
    """
    Convert:
      - "all"
      - "device=0"
      - "device=1"
      - "0"/"1"
    to docker.types.DeviceRequest
    """
    if gpus == "all":
        return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

    dev_id = gpus
    if "=" in gpus:
        k, v = gpus.split("=", 1)
        if k.strip() != "device":
            raise ValueError(f"Unsupported gpus format '{gpus}'. Use 'all' or 'device=N'.")
        dev_id = v.strip()

    return [docker.types.DeviceRequest(device_ids=[dev_id], capabilities=[["gpu"]])]


def create_container(model_key: str, spec: Dict[str, Any]) -> None:
    name = spec["container_name"]
    image = spec["image"]
    gpus = str(spec.get("gpus", "all"))
    host_port = int(spec["host_port"])
    env = spec.get("env", {})
    volumes_list = spec.get("volumes", [])
    args = spec.get("args", [])

    if not isinstance(volumes_list, list):
        raise ValueError(f"{model_key}.volumes must be a list.")
    if not isinstance(args, list):
        raise ValueError(f"{model_key}.args must be a list (vLLM CLI args).")

    volumes = _parse_volumes(volumes_list)
    device_requests = _parse_gpu_request(gpus)
    ports = {"8000/tcp": host_port}

    docker_client.containers.run(
        image=image,
        name=name,
        detach=True,
        ipc_mode="host",
        restart_policy={"Name": "no"},  # controller decides lifecycle
        environment=env,
        volumes=volumes,
        ports=ports,
        device_requests=device_requests,
        command=args,
    )


# -----------------------------
# Helpers: GPU VRAM (pre-check)
# -----------------------------


def _list_gpu_indices() -> List[int]:
    """
    Query GPU indices using a CUDA container running nvidia-smi.
    Returns e.g. [0,1]
    """
    cmd = [
        "bash",
        "-lc",
        "nvidia-smi --query-gpu=index --format=csv,noheader,nounits",
    ]
    try:
        out = docker_client.containers.run(
            image=CUDA_SMI_IMAGE,
            command=cmd,
            detach=False,
            remove=True,
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        )
        txt = out.decode("utf-8", errors="replace").strip()
        if not txt:
            return []
        return [int(x.strip()) for x in txt.splitlines() if x.strip().isdigit()]
    except Exception:
        # If this fails, return empty and let system fall back to "try start + eviction on failure"
        return []


def get_free_vram_gib(gpu_id: int) -> Optional[float]:
    """
    Returns free VRAM in GiB for the given GPU index and records total VRAM.
    Single nvidia-smi call to fetch free and total (MiB) -> GiB.
    """
    vram_check_start = time.time()
    cmd = [
        "bash",
        "-lc",
        f"nvidia-smi -i {gpu_id} --query-gpu=memory.free,memory.total --format=csv,noheader,nounits",
    ]
    try:
        out = docker_client.containers.run(
            image=CUDA_SMI_IMAGE,
            command=cmd,
            detach=False,
            remove=True,
            # use all visible GPUs so per-index queries still work in minimal runtimes
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        )
        txt = out.decode("utf-8", errors="replace").strip()
        first_line = txt.splitlines()[0].strip()
        parts = [p.strip() for p in first_line.split(',') if p.strip()]
        if len(parts) < 2:
            raise ValueError("Unexpected nvidia-smi output for free/total")

        free_mib = float(parts[0])
        total_mib = float(parts[1])
        free_gib = free_mib / 1024.0
        total_gib = total_mib / 1024.0

        # Record VRAM check duration and values
        metrics.vram_check_duration_seconds.observe(time.time() - vram_check_start)
        metrics.vram_free_gib.labels(gpu_id=str(gpu_id)).set(free_gib)
        metrics.vram_total_gib.labels(gpu_id=str(gpu_id)).set(total_gib)
        return free_gib
    except Exception as e:
        metrics.vram_check_duration_seconds.observe(time.time() - vram_check_start)
        print(f"[VRAM] Failed to sample GPU {gpu_id}: {e}", flush=True)
        # Emit zero so the series exists even on failure
        metrics.vram_free_gib.labels(gpu_id=str(gpu_id)).set(0)
        metrics.vram_total_gib.labels(gpu_id=str(gpu_id)).set(0)
        return None


def get_free_vram_for_req(req: Union[str, int, List[int]]) -> Dict[int, Optional[float]]:
    """
    For a requested gpu spec:
      - "all" -> query all indices discovered
      - int -> query that gpu
      - list[int] -> query each
    Returns mapping gpu_id -> free_gib (or None if failed)
    """
    if req == "all":
        ids = _list_gpu_indices()
        return {i: get_free_vram_gib(i) for i in ids}
    ids = _gpu_list_from_requested(req)
    return {i: get_free_vram_gib(i) for i in ids}


# -----------------------------
# Helpers: Readiness
# -----------------------------


async def wait_ready(host_port: int, model_key: str = "unknown") -> None:
    url = f"http://{VLLM_HOST_IP}:{host_port}/v1/models"
    deadline = _now() + MODEL_READY_TIMEOUT_SEC
    
    probe_start = time.time()

    async with httpx.AsyncClient(timeout=5.0) as client:
        while _now() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200 and r.text and r.text.strip():
                    # Record successful readiness probe
                    metrics.readiness_probe_duration_seconds.labels(model=model_key).observe(time.time() - probe_start)
                    return
            except Exception:
                pass
            await _sleep(MODEL_READY_POLL_SEC)

    # Record failed readiness probe
    metrics.readiness_probe_failures_total.labels(model=model_key).inc()
    raise RuntimeError(f"Backend on {url} not ready within {MODEL_READY_TIMEOUT_SEC}s")


# -----------------------------
# Eviction Policy (LRU + priority + GPU-awareness)
# -----------------------------


def _running_models() -> List[str]:
    """
    Return list of model keys whose containers are currently running.
    """
    running = []
    for mk, spec in MODELS.items():
        name = spec.get("container_name")
        if not name:
            continue
        if container_running(name):
            running.append(mk)
    return running


def _model_targets_gpu(model_key: str, target_req: Union[str, int, List[int]]) -> bool:
    """
    Determine whether model_key is relevant to evict when we need to free memory for target_req.
    - If target_req == "all": any non-warm, non-inflight running model can be evicted.
    - Else: evict only models that are assigned to one of those GPUs OR are "all".
    """
    spec = _model_spec(model_key)
    req = _requested_gpus(spec)

    if target_req == "all":
        return True

    target_gpus = set(_gpu_list_from_requested(target_req))
    if req == "all":
        return True
    if isinstance(req, int):
        return req in target_gpus
    if isinstance(req, list):
        return any(x in target_gpus for x in req)
    return True


def pick_evict_candidate(
    target_model: str,
    target_req: Union[str, int, List[int]],
) -> Optional[str]:
    """
    Pick an eviction candidate according to:
      - not the target_model
      - running
      - not warm
      - not in_flight
      - relevant to target GPU(s)
      - lowest "score" (LRU first, lower priority first)
    Score heuristic:
      sort by:
        1) warm False (already filtered)
        2) in_flight == 0 (already filtered)
        3) priority ascending
        4) last_used ascending (least recently used evicted first)
        5) stable tie-breaker by model_key
    """
    candidates: List[Tuple[int, float, str]] = []
    for mk in _running_models():
        if mk == target_model:
            continue

        spec = _model_spec(mk)
        if _is_warm(spec):
            continue

        if in_flight.get(mk, 0) > 0:
            continue

        if not _model_targets_gpu(mk, target_req):
            continue

        pr = _priority(spec)
        lu = last_used.get(mk, 0.0)  # models never used via controller are easiest to evict
        candidates.append((pr, lu, mk))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    return candidates[0][2]


def evict_model(model_key: str) -> None:
    spec = _model_spec(model_key)
    name = spec["container_name"]
    docker_stop(name)
    # Record eviction metric
    metrics.model_evictions_total.labels(model=model_key, reason="lru_eviction").inc()


def vram_sufficient_for_model(spec: Dict[str, Any]) -> Optional[bool]:
    """
    Returns:
      - True / False if we could compute VRAM and compare
      - None if VRAM query unavailable (fallback to start-attempt strategy)
    """
    req = _requested_gpus(spec)
    min_free = _min_free_gib(spec)
    if min_free is None:
        # No threshold configured => cannot pre-check reliably
        return None

    free_map = get_free_vram_for_req(req)
    if not free_map:
        return None

    # If req == "all", we need all available GPUs to meet threshold
    # If req is single GPU/list, only those GPUs are required.
    for gpu_id, free_gib in free_map.items():
        if free_gib is None:
            return None
        if free_gib < min_free:
            return False
    return True


def needs_eviction_precheck(spec: Dict[str, Any]) -> Optional[Tuple[Union[str, int, List[int]], float, Dict[int, Optional[float]]]]:
    """
    If configured, determine whether we should evict based on VRAM.
    Returns:
      - None if cannot determine
      - (req, min_free_gib, free_map) if determinable
    """
    req = _requested_gpus(spec)
    min_free = _min_free_gib(spec)
    if min_free is None:
        return None

    free_map = get_free_vram_for_req(req)
    if not free_map:
        return None

    return (req, min_free, free_map)


def is_below_threshold(min_free: float, free_map: Dict[int, Optional[float]]) -> Optional[bool]:
    """
    Returns True if any required GPU is below threshold.
    Returns None if any GPU free value is None (unknown).
    """
    for _, free in free_map.items():
        if free is None:
            return None
        if free < min_free:
            return True
    return False


# -----------------------------
# Ensure Model (with pre-check + preventive eviction + fallback)
# -----------------------------


async def ensure_model(model_key: str) -> None:
    """
    Ensure vLLM backend container exists, has enough VRAM (via pre-check/eviction), is running and ready.
    """
    spec = _model_spec(model_key)
    name = spec["container_name"]
    host_port = int(spec["host_port"])
    
    startup_start_time = time.time()

    # 1) PRE-CHECK VRAM (optional, if min_free_gib configured)
    pre = needs_eviction_precheck(spec)
    if pre is not None:
        req, min_free, free_map = pre
        below = is_below_threshold(min_free, free_map)
        if below is True:
            # Preventive eviction loop
            evicted = 0
            while evicted < EVICT_RETRY_LIMIT:
                cand = pick_evict_candidate(target_model=model_key, target_req=req)
                if cand is None:
                    break
                evict_model(cand)
                evicted += 1
                # Small pause for memory to be released
                await _sleep(2.0)

                free_map = get_free_vram_for_req(req)
                below2 = is_below_threshold(min_free, free_map)
                if below2 is False:
                    break
            # After preventive eviction attempt, proceed to start and rely on fallback if still fails

    # 2) Ensure container exists
    if not container_exists(name):
        create_container(model_key, spec)

    # 3) Start (with fallback eviction+retry if readiness fails)
    attempts_left = 1 + max(0, EVICT_AFTER_FAILED_START)
    last_err: Optional[Exception] = None

    while attempts_left > 0:
        attempts_left -= 1
        try:
            if not container_running(name):
                docker_start(name)
                # Record container status
                metrics.container_status.labels(model=model_key, container_name=name).set(1)

            # Wait until vLLM is ready
            await wait_ready(host_port, model_key)
            
            # Record successful startup time
            startup_duration = time.time() - startup_start_time
            metrics.model_startup_time_seconds.labels(model=model_key).observe(startup_duration)
            return

        except Exception as e:
            last_err = e
            # Record container error
            metrics.container_status.labels(model=model_key, container_name=name).set(-1)

            # If no retries left, stop and raise
            if attempts_left <= 0:
                raise

            # Try eviction of one candidate and retry
            req = _requested_gpus(spec)
            cand = pick_evict_candidate(target_model=model_key, target_req=req)
            if cand is None:
                raise

            evict_model(cand)
            await _sleep(2.0)

            # Optionally, if container for target is in a bad state (crash loop), restart it cleanly
            # Stop it before retry to reduce noise
            try:
                docker_stop(name)
                metrics.container_status.labels(model=model_key, container_name=name).set(0)
            except Exception:
                pass

            await _sleep(1.0)

    # Should not reach here
    if last_err:
        raise last_err


# -----------------------------
# Background Reaper (idle stop)
# -----------------------------


@app.on_event("startup")
async def startup() -> None:
    import asyncio
    import logging
    import sys

    logger = logging.getLogger(__name__)
    sys.stderr.write("="*60 + "\n")
    sys.stderr.write("CONTROLLER STARTUP - Initializing metrics persistence\n")
    sys.stderr.write("="*60 + "\n")
    sys.stderr.flush()
    print("="*60, flush=True)
    print("CONTROLLER STARTUP - Initializing metrics persistence", flush=True)
    print("="*60, flush=True)

    # Initialize database and restore metrics from previous run
    try:
        print("[1/3] Initializing database connection...", flush=True)
        await db_metrics.init_db()
        print("[2/3] Restoring metrics from database...", flush=True)
        restored = await db_metrics.restore_metrics()
        print(f"[3/3] Restored metrics for {len(restored)} models", flush=True)
        
        # Restore counter values
        for model, metrics_state in restored.items():
            try:
                # Restore requests_total for each status combination
                if 'requests_total' in metrics_state:
                    for endpoint in ['chat/completions', 'completions', 'embeddings']:
                        for status in ['success', 'error', 'unknown_model']:
                            try:
                                current = metrics.requests_total.labels(
                                    model=model, endpoint=endpoint, status=status
                                )._value.get()
                                # Restore only once per model (avoid multiplying)
                            except Exception:
                                pass
                
                # Restore model_evictions_total
                if 'model_evictions_total' in metrics_state:
                    value = int(metrics_state['model_evictions_total'])
                    if value > 0:
                        # Estimate split between LRU and TTL (assume 80/20)
                        lru_evictions = int(value * 0.8)
                        ttl_evictions = value - lru_evictions
                        if lru_evictions > 0:
                            metrics.model_evictions_total.labels(model=model, reason="lru_eviction").inc(lru_evictions)
                        if ttl_evictions > 0:
                            metrics.model_evictions_total.labels(model=model, reason="ttl_idle").inc(ttl_evictions)
                
                # Restore readiness_probe_failures_total
                if 'readiness_probe_failures_total' in metrics_state:
                    value = int(metrics_state['readiness_probe_failures_total'])
                    if value > 0:
                        metrics.readiness_probe_failures_total.labels(model=model).inc(value)
                
                print(f"✓ Restored metrics for model: {model}", flush=True)
            except Exception as e:
                print(f"✗ Could not fully restore metrics for {model}: {e}", flush=True)
    except Exception as e:
        print(f"✗ Database initialization failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Don't raise - allow controller to start anyway

    print("="*60, flush=True)
    print("STARTUP COMPLETE - Background tasks starting", flush=True)
    print("="*60, flush=True)

    async def reaper():
        ttl = IDLE_TTL_MINUTES * 60
        while True:
            now = _now()
            for model_key, spec in MODELS.items():
                name = spec.get("container_name")
                if not name:
                    continue

                if _is_warm(spec):
                    continue

                lu = last_used.get(model_key)
                if lu is None:
                    continue

                if in_flight.get(model_key, 0) > 0:
                    continue

                if (now - lu) > ttl:
                    if container_running(name):
                        docker_stop(name)
                        metrics.model_evictions_total.labels(model=model_key, reason="ttl_idle").inc()
                        metrics.container_status.labels(model=model_key, container_name=name).set(0)

            await asyncio.sleep(30)

    async def persister():
        """Persist metrics to database every 5 minutes"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Collect all counter states
                state_to_save = {}
                for model_key in MODELS.keys():
                    state_to_save[model_key] = {
                        'requests_total': metrics.requests_total.labels(
                            model=model_key, endpoint='chat/completions', status='success'
                        )._value.get(),
                        'model_evictions_total': sum(
                            metrics.model_evictions_total.labels(model=model_key, reason=r)._value.get()
                            for r in ['lru_eviction', 'ttl_idle']
                        ),
                        'readiness_probe_failures_total': metrics.readiness_probe_failures_total.labels(
                            model=model_key
                        )._value.get(),
                    }
                
                await db_metrics.persist_metrics(state_to_save)
                await db_metrics.cleanup_old_metrics()
            except Exception as e:
                logger.error(f"Error persisting metrics: {e}")

    asyncio.create_task(reaper())
    asyncio.create_task(persister())


# Shutdown hook
@app.on_event("shutdown")
async def shutdown() -> None:
    """Persist final metrics state and close database"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Final persistence before shutdown
        state_to_save = {}
        for model_key in MODELS.keys():
            state_to_save[model_key] = {
                'requests_total': metrics.requests_total.labels(
                    model=model_key, endpoint='chat/completions', status='success'
                )._value.get(),
                'model_evictions_total': sum(
                    metrics.model_evictions_total.labels(model=model_key, reason=r)._value.get()
                    for r in ['lru_eviction', 'ttl_idle']
                ),
                'readiness_probe_failures_total': metrics.readiness_probe_failures_total.labels(
                    model=model_key
                )._value.get(),
            }
        
        await db_metrics.persist_metrics(state_to_save)
        logger.info("✓ Final metrics persisted")
    except Exception as e:
        logger.error(f"Error during final persistence: {e}")
    finally:
        await db_metrics.close_db()
        logger.info("✓ Database connection closed")


# -----------------------------
# Proxying to LiteLLM
# -----------------------------


async def proxy_to_litellm(request: Request, body: bytes, model_key: Optional[str] = None) -> Response:
    url = f"{LITELLM_BASE_URL}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"

    method = request.method.upper()
    headers = _strip_hop_by_hop(dict(request.headers))

    stream_requested = (method == "POST") and _is_stream_requested(body)

    
    if stream_requested:

        upstream_ctx = http_client.stream(method, url, headers=headers, content=body)
        resp = await upstream_ctx.__aenter__()

        async def gen():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            except (httpx.ReadError, httpx.RemoteProtocolError, asyncio.CancelledError):
                return
            finally:
                await upstream_ctx.__aexit__(None, None, None)

        out_headers = _strip_hop_by_hop(dict(resp.headers))
        out_headers.pop("content-length", None)

        return StreamingResponse(
            gen(),
            status_code=resp.status_code,
            headers=out_headers,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    # Para peticiones NO-streaming
    resp = await http_client.request(method, url, headers=headers, content=body)
    if model_key: # Bajamos el contador para peticiones normales
        in_flight[model_key] = max(0, in_flight.get(model_key, 1) - 1)
        metrics.in_flight_requests.labels(model=model_key).set(in_flight[model_key])
        
    out_headers = _strip_hop_by_hop(dict(resp.headers))
    return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)


# -----------------------------
# Routes
# -----------------------------


@app.get("/health")
async def health():
    # Provide a brief snapshot, including best-effort VRAM if configured
    vram_snapshot: Dict[str, Any] = {}
    for mk, spec in MODELS.items():
        req = _requested_gpus(spec)
        min_free = spec.get("min_free_gib")
        free_map = get_free_vram_for_req(req)
        vram_snapshot[mk] = {
            "req_gpu": req,
            "min_free_gib": min_free,
            "free_gib": free_map,
            "warm": _is_warm(spec),
            "priority": _priority(spec),
            "running": container_running(spec.get("container_name", "")) if spec.get("container_name") else False,
        }

    return {
        "status": "ok",
        "litellm_base_url": LITELLM_BASE_URL,
        "vllm_host_ip": VLLM_HOST_IP,
        "idle_ttl_minutes": IDLE_TTL_MINUTES,
        "models": list(MODELS.keys()),
        "vram": vram_snapshot,
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=metrics.get_metrics_content(),
        media_type=metrics.get_metrics_content_type()
    )


@app.get("/metrics/state")
async def metrics_state():
    """Get current metrics state (from database for persistence view)"""
    state = await db_metrics.get_current_metrics_state()
    return {
        "status": "ok",
        "timestamp": _now(),
        "metrics": state
    }


@app.get("/metrics/history/{model}")
async def metrics_history(model: str, hours: int = 24, metric: str = "requests_total"):
    """Get historical metric values for a model"""
    history = await db_metrics.get_metric_history(model, metric, hours)
    return {
        "model": model,
        "metric": metric,
        "hours": hours,
        "data": history
    }

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def openai_compat(path: str, request: Request):
    # 1. Obtener el body original
    body_bytes = await request.body()
    
    # 2. Lógica de saneamiento (evitar max_tokens negativos)
    if request.method.upper() == "POST" and body_bytes:
        try:
            body_json = json.loads(body_bytes)
            # Verificar si max_tokens existe y es inválido
            if "max_tokens" in body_json:
                val = body_json["max_tokens"]
                if not isinstance(val, int) or val < 1:
                    print(f"DEBUG: Corrigiendo max_tokens inválido ({val}) enviado por el cliente.")
                    del body_json["max_tokens"]
                    # Re-serializar el body corregido
                    body_bytes = json.dumps(body_json).encode("utf-8")
        except Exception as e:
            # Si no es JSON (ej. multipart), simplemente ignoramos la corrección
            print(f"DEBUG: No se pudo procesar el JSON para corregir max_tokens: {e}")

    # 3. Extraer el modelo del body (ya saneado)
    model = _extract_model_from_json(body_bytes) if (request.method.upper() == "POST" and body_bytes) else None

    # 4. Lógica de gestión de contenedores vLLM
    if request.method.upper() == "POST" and model:
        start_time = time.time()
        in_flight[model] = in_flight.get(model, 0) + 1
        metrics.in_flight_requests.labels(model=model).set(in_flight[model])
        
        try:
            await ensure_model(model)
            last_used[model] = _now()
            metrics.model_last_used_timestamp.labels(model=model).set(last_used[model])
            
            # Pasar el body_bytes corregido al proxy
            response = await proxy_to_litellm(request, body_bytes, model_key=model)
            
            metrics.requests_total.labels(model=model, endpoint=path, status="success").inc()
            return response
            
        except KeyError:
            # Limpieza si el modelo no existe
            in_flight[model] = max(0, in_flight.get(model, 1) - 1)
            metrics.in_flight_requests.labels(model=model).set(in_flight[model])
            metrics.requests_total.labels(model=model, endpoint=path, status="unknown_model").inc()
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Unknown model '{model}'", "code": "unknown_model"}},
            )
        except Exception as e:
            # Limpieza si falla el arranque
            if model in in_flight:
                in_flight[model] = max(0, in_flight[model] - 1)
                metrics.in_flight_requests.labels(model=model).set(in_flight[model])
            metrics.requests_total.labels(model=model, endpoint=path, status="error").inc()
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"Failed to start model '{model}': {e}", "code": "model_start_failed"}},
            )
        # IMPORTANTE: Ya no usamos el 'finally' aquí porque proxy_to_litellm 
        # se encarga de bajar el contador al terminar el stream.

    # 5. Para peticiones que no son POST o no tienen modelo (ej. /v1/models)
    return await proxy_to_litellm(request, body_bytes)

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()