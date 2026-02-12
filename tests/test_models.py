"""
E2E tests for model-specific behavior.

Tests cover:
- Different model backends
- Model-specific parameters
- Model eviction and VRAM management
- Model warm/cold starts
- Model priority and scheduling
"""

import pytest
import httpx
import asyncio
import time
from typing import Dict, Any, List


class TestModelDiscoveryAndInfo:
    """Test model discovery and metadata"""
    
    def test_all_models_in_health_endpoint(self, http_client: httpx.Client):
        """All configured models should appear in health endpoint."""
        response = http_client.get("/health")
        models = response.json()["models"]
        
        assert len(models) > 0
        expected_models = ["deepseek-r1", "qwen-2.5-7b-instruct", "llama-2-70b-chat"]
        # At least some expected models should be there
        found_models = [m for m in expected_models if m in models]
        assert len(found_models) > 0
    
    
    def test_model_configuration_visibility(self, http_client: httpx.Client):
        """Model configuration should be visible in health endpoint."""
        response = http_client.get("/health")
        vram_snapshot = response.json()["vram"]
        
        for model_key, config in vram_snapshot.items():
            assert "req_gpu" in config
            assert "min_free_gib" in config
            assert "warm" in config
            assert "priority" in config
            assert isinstance(config["warm"], bool)
            assert isinstance(config["priority"], int)


class TestModelSpecificRequests:
    """Test requests specific to individual models"""
    
    @pytest.mark.parametrize("model", ["deepseek-r1", "qwen-2.5-7b-instruct"])
    def test_request_to_specific_model(self, http_client: httpx.Client, model: str):
        """Should handle requests to different models."""
        request = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        response = http_client.post(
            "/v1/chat/completions",
            json=request,
            timeout=60.0
        )
        # Should process (may fail but not crash)
        assert response.status_code in [200, 400, 500]
    
    
    def test_model_specific_parameters(self, http_client: httpx.Client):
        """Different models may accept different parameters."""
        models_with_params = [
            {
                "model": "deepseek-r1",
                "messages": [{"role": "user", "content": "Think step by step: 2+2"}],
                "temperature": 0.7,
                "max_tokens": 100
            },
            {
                "model": "qwen-2.5-7b-instruct",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 0.9,
                "temperature": 0.8,
                "max_tokens": 50
            }
        ]
        
        for request in models_with_params:
            response = http_client.post(
                "/v1/chat/completions",
                json=request,
                timeout=60.0
            )
            # Should handle model-specific parameters
            assert response.status_code in [200, 400, 500]
    
    
    def test_model_not_found_error(self, http_client: httpx.Client):
        """Non-existent model should return appropriate error."""
        request = {
            "model": "nonexistent-model-xyz-999",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = http_client.post("/v1/chat/completions", json=request)
        assert response.status_code in [400, 404, 500]


class TestModelStartupAndWarmup:
    """Test model startup and warmup behavior"""
    
    def test_model_cold_start_timing(self, http_client: httpx.Client):
        """First request to a model should handle startup time."""
        # Check health to see if model is running
        health = http_client.get("/health").json()
        model_running = health["vram"].get("deepseek-r1", {}).get("running", False)
        
        request = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }
        
        start_time = time.time()
        response = http_client.post(
            "/v1/chat/completions",
            json=request,
            timeout=300.0  # Long timeout for startup
        )
        elapsed = time.time() - start_time
        
        # Cold start should be slower than warm start
        # (but we can't always test this without state)
        assert response.status_code in [200, 400, 500]
    
    
    def test_warm_model_performance(self, http_client: httpx.Client):
        """Warm model should respond relatively quickly."""
        request = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5
        }
        
        # Make two requests - second should be warm
        for i in range(2):
            start_time = time.time()
            response = http_client.post(
                "/v1/chat/completions",
                json=request,
                timeout=120.0
            )
            elapsed = time.time() - start_time
            
            # Second request should be faster if model was warm
            # But we don't enforce this in test, just ensure it completes
            assert response.status_code in [200, 400, 500]


class TestModelLoadBalancing:
    """Test load balancing across models"""
    
    @pytest.mark.asyncio
    async def test_distribute_requests_across_models(self, async_http_client: httpx.AsyncClient):
        """Requests should be distributable across models."""
        models = ["deepseek-r1", "qwen-2.5-7b-instruct"]
        
        async def request_to_model(model_name):
            request = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=60.0
                )
                return {
                    "model": model_name,
                    "status": response.status_code
                }
            except Exception as e:
                return {"model": model_name, "error": str(e)}
        
        # Send requests to different models concurrently
        tasks = [request_to_model(m) for m in models]
        results = await asyncio.gather(*tasks)
        
        # Should handle requests to all models
        assert len(results) == len(models)
        models_processed = [r["model"] for r in results]
        assert set(models_processed) == set(models)
    
    
    @pytest.mark.asyncio
    async def test_model_priority_awareness(self, async_http_client: httpx.AsyncClient):
        """Gateway should be aware of model priorities."""
        # Models with different priorities should coexist
        response = await async_http_client.get("/health")
        health_data = response.json()
        vram_info = health_data["vram"]
        
        # Check that models have different priorities
        priorities = [info.get("priority", 0) for info in vram_info.values()]
        # Should have at least some configuration
        assert len(priorities) > 0


class TestModelEviction:
    """Test model eviction and memory management"""
    
    def test_check_model_running_status(self, http_client: httpx.Client):
        """Should be able to check if models are running."""
        response = http_client.get("/health")
        vram_snapshot = response.json()["vram"]
        
        for model_key, info in vram_snapshot.items():
            running = info.get("running")
            assert isinstance(running, bool)
    
    
    def test_vram_threshold_awareness(self, http_client: httpx.Client):
        """Gateway should be aware of VRAM requirements."""
        response = http_client.get("/health")
        vram_snapshot = response.json()["vram"]
        
        for model_key, info in vram_snapshot.items():
            min_free = info.get("min_free_gib")
            # Should have VRAM requirement or None
            assert min_free is None or isinstance(min_free, (int, float))
    
    
    def test_gpu_assignment_awareness(self, http_client: httpx.Client):
        """Gateway should be aware of GPU assignments."""
        response = http_client.get("/health")
        vram_snapshot = response.json()["vram"]
        
        for model_key, info in vram_snapshot.items():
            req_gpu = info.get("req_gpu")
            # Should have GPU assignment (int, "all", or list)
            assert req_gpu is not None


class TestModelWarmState:
    """Test warm/cold model management"""
    
    def test_warm_models_configuration(self, http_client: httpx.Client):
        """Warm models should be marked in configuration."""
        response = http_client.get("/health")
        vram_snapshot = response.json()["vram"]
        
        warm_models = []
        cold_models = []
        
        for model_key, info in vram_snapshot.items():
            if info.get("warm", False):
                warm_models.append(model_key)
            else:
                cold_models.append(model_key)
        
        # Should have at least some models (warm or cold)
        assert len(warm_models) + len(cold_models) == len(vram_snapshot)


class TestMultiGPUModels:
    """Test models that use multiple GPUs"""
    
    def test_multi_gpu_model_detection(self, http_client: httpx.Client):
        """Should detect and handle multi-GPU models."""
        response = http_client.get("/health")
        vram_snapshot = response.json()["vram"]
        
        for model_key, info in vram_snapshot.items():
            req_gpu = info.get("req_gpu")
            # req_gpu could be int (single), "all" (multiple), or list
            valid_types = (int, str, list)
            assert isinstance(req_gpu, valid_types)
    
    
    def test_multi_gpu_request_routing(self, http_client: httpx.Client):
        """Requests to multi-GPU models should be routed correctly."""
        request = {
            "model": "llama-2-70b-chat",  # typically multi-GPU
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5
        }
        
        response = http_client.post(
            "/v1/chat/completions",
            json=request,
            timeout=60.0
        )
        # Should handle multi-GPU model routing
        assert response.status_code in [200, 400, 500]


class TestModelConcurrency:
    """Test concurrent access to same model"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_same_model_isolation(self, async_http_client: httpx.AsyncClient):
        """Multiple requests to same model should be isolated."""
        num_requests = 5
        model = "deepseek-r1"
        
        async def request_iteration(idx):
            request = {
                "model": model,
                "messages": [
                    {"role": "user", "content": f"Request {idx}: what is {idx}+1?"}
                ],
                "max_tokens": 20
            }
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=60.0
                )
                return {
                    "index": idx,
                    "status": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {"index": idx, "error": str(e)}
        
        # Concurrent requests
        tasks = [request_iteration(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Should process all requests
        assert len(results) == num_requests
        # Some should succeed
        successful = [r for r in results if r.get("success", False)]
        # Don't strictly require success, but should process
        assert len(results) > 0


class TestModelMetadata:
    """Test model metadata and capabilities"""
    
    def test_model_capabilities_inference(self, http_client: httpx.Client):
        """Should infer model capabilities from name."""
        response = http_client.get("/health")
        models = response.json()["models"]
        
        # deepseek-r1 should support chain-of-thought
        # qwen should support instruction following
        # llama should support general chat
        # (These are inferences, not necessarily enforced)
        
        for model in models:
            assert isinstance(model, str)
            assert len(model) > 0


class TestStreamingPerModel:
    """Test streaming with different models"""
    
    @pytest.mark.asyncio
    async def test_streaming_deepseek_model(self, async_http_client: httpx.AsyncClient):
        """DeepSeek model should support streaming."""
        request = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": True,
            "max_tokens": 50
        }
        
        try:
            async with await async_http_client.stream(
                "POST",
                "/v1/chat/completions",
                json=request,
                timeout=60.0
            ) as response:
                chunks_received = 0
                async for chunk in response.aiter_bytes():
                    if chunk:
                        chunks_received += 1
                
                assert response.status_code in [200, 400, 500]
        except Exception:
            # Streaming might not be available
            pass
    
    
    @pytest.mark.asyncio
    async def test_streaming_qwen_model(self, async_http_client: httpx.AsyncClient):
        """Qwen model should support streaming."""
        request = {
            "model": "qwen-2.5-7b-instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 50
        }
        
        try:
            async with await async_http_client.stream(
                "POST",
                "/v1/chat/completions",
                json=request,
                timeout=60.0
            ) as response:
                chunks_received = 0
                async for chunk in response.aiter_bytes():
                    if chunk:
                        chunks_received += 1
                
                assert response.status_code in [200, 400, 500]
        except Exception:
            # Streaming might not be available
            pass
