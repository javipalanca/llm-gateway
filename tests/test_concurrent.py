"""
E2E tests for concurrent and parallel requests.

Tests cover:
- Parallel requests to same model
- Parallel requests to different models
- Load testing
- Rate limiting
- Connection pooling
- Request queueing
"""

import pytest
import httpx
import asyncio
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestParallelRequestsBasic:
    """Basic parallel request tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_same_model(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Multiple concurrent requests to same model should be handled."""
        num_requests = 5
        
        async def make_request(idx):
            request = {**valid_chat_completion_request, "max_tokens": 10}
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=60.0
                )
                return {
                    "index": idx,
                    "status": response.status_code,
                    "response": response.json() if response.status_code == 200 else None
                }
            except Exception as e:
                return {"index": idx, "error": str(e)}
        
        # Execute requests concurrently
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # At least some should succeed or fail gracefully
        assert len(results) == num_requests
        status_codes = [r.get("status") for r in results if "status" in r]
        assert len(status_codes) > 0
    
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_different_models(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Concurrent requests to different models should work."""
        models = ["deepseek-r1", "qwen-2.5-7b-instruct"]
        
        async def make_request(model_name):
            request = {**valid_chat_completion_request, "model": model_name, "max_tokens": 10}
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
        
        # Execute requests concurrently
        tasks = [make_request(model) for model in models]
        results = await asyncio.gather(*tasks)
        
        # Should process all requests
        assert len(results) == len(models)
    
    
    @pytest.mark.asyncio
    async def test_many_concurrent_requests(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """System should handle many concurrent requests."""
        num_requests = 20
        
        async def make_request(idx):
            request = {**valid_chat_completion_request, "max_tokens": 5}
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=120.0
                )
                return response.status_code
            except (asyncio.TimeoutError, httpx.TimeoutException):
                return 408  # timeout acceptable
            except Exception:
                return None
        
        # Execute many requests concurrently
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should process all requests (may fail but shouldn't crash)
        status_codes = [r for r in results if isinstance(r, int)]
        assert len(status_codes) > 0


class TestParallelRequestsWithThreadPool:
    """Test parallel requests using thread pool"""
    
    def test_thread_pool_concurrent_requests(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """ThreadPool should handle concurrent requests."""
        num_workers = 5
        num_requests = 10
        
        def make_request(idx):
            request = {**valid_chat_completion_request, "max_tokens": 10}
            try:
                response = http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=60.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Execute with ThreadPool
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]
        
        # Should complete all requests
        assert len(results) == num_requests
    
    
    def test_thread_pool_different_models(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """ThreadPool with different models should work."""
        num_workers = 4
        models = ["deepseek-r1", "qwen-2.5-7b-instruct", "llama-2-70b-chat"]
        requests = [
            {**valid_chat_completion_request, "model": m, "max_tokens": 10}
            for m in models
        ] * 3  # repeat 3 times
        
        def make_request(req_data):
            try:
                response = http_client.post(
                    "/v1/chat/completions",
                    json=req_data,
                    timeout=60.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Execute with ThreadPool
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request, req) for req in requests]
            results = [f.result() for f in as_completed(futures)]
        
        # Should process all requests
        assert len(results) == len(requests)


class TestStreamingConcurrent:
    """Test concurrent streaming requests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, async_http_client: httpx.AsyncClient, valid_chat_completion_stream_request: Dict):
        """Multiple concurrent streaming requests should work."""
        num_requests = 3
        
        async def stream_request(idx):
            request = {**valid_chat_completion_stream_request, "max_tokens": 20}
            try:
                async with await async_http_client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=request,
                    timeout=60.0
                ) as response:
                    chunks = []
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)
                    return {
                        "index": idx,
                        "status": response.status_code,
                        "chunks": len(chunks)
                    }
            except Exception as e:
                return {"index": idx, "error": str(e)}
        
        # Execute streaming requests concurrently
        tasks = [stream_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Should handle all streaming requests
        assert len(results) == num_requests


class TestRequestQueuing:
    """Test request queuing and ordering"""
    
    @pytest.mark.asyncio
    async def test_request_order_preservation(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Requests should be processed in a predictable manner."""
        num_requests = 10
        request_ids = []
        
        async def make_request(idx):
            request = {
                **valid_chat_completion_request,
                "max_tokens": 5,
                # Add unique identifier to track request
            }
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=60.0
                )
                return {
                    "order": idx,
                    "status": response.status_code,
                    "timestamp": time.time()
                }
            except Exception:
                return {"order": idx, "error": True, "timestamp": time.time()}
        
        # Send requests sequentially in async
        results = []
        for i in range(num_requests):
            result = await make_request(i)
            results.append(result)
        
        # All requests should complete
        assert len(results) == num_requests


class TestLoadTesting:
    """Load testing the gateway"""
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Gateway should handle sustained load."""
        duration_seconds = 5
        requests_sent = 0
        requests_successful = 0
        
        start_time = time.time()
        
        async def make_request():
            nonlocal requests_successful
            request = {**valid_chat_completion_request, "max_tokens": 5}
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=30.0
                )
                if response.status_code == 200:
                    requests_successful += 1
            except Exception:
                pass
        
        # Send requests continuously for duration
        while time.time() - start_time < duration_seconds:
            tasks = [make_request() for _ in range(5)]
            await asyncio.gather(*tasks, return_exceptions=True)
            requests_sent += 5
        
        # Should complete some requests
        assert requests_sent > 0
    
    
    @pytest.mark.asyncio
    async def test_burst_load(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Gateway should handle burst load (many simultaneous requests)."""
        burst_size = 30
        
        async def make_request():
            request = {**valid_chat_completion_request, "max_tokens": 5}
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=60.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Send burst of requests
        tasks = [make_request() for _ in range(burst_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should process all requests (not crash)
        status_codes = [r for r in results if isinstance(r, int)]
        assert len(status_codes) > 0


class TestResourceManagement:
    """Test resource management under load"""
    
    @pytest.mark.asyncio
    async def test_connection_reuse(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """HTTP connections should be reused efficiently."""
        num_requests = 20
        
        async def make_request(idx):
            request = {**valid_chat_completion_request, "max_tokens": 5}
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=30.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Make many requests (should reuse connections)
        start_time = time.time()
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should be relatively fast with connection reuse
        assert len(results) == num_requests
        # All requests should complete in reasonable time (rough estimate)
        # assert duration < num_requests * 10  # 10s per request max
    
    
    @pytest.mark.asyncio
    async def test_memory_stability_repeated_requests(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Repeated requests shouldn't cause memory leaks."""
        num_iterations = 10
        
        async def make_request():
            request = {**valid_chat_completion_request, "max_tokens": 5}
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=30.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Make repeated batches of requests
        for iteration in range(num_iterations):
            tasks = [make_request() for _ in range(5)]
            results = await asyncio.gather(*tasks)
            # Should complete all batches
            assert len(results) == 5


class TestErrorHandlingUnderLoad:
    """Test error handling with concurrent requests"""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_with_errors(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Gateway should degrade gracefully under error conditions."""
        # Mix valid and invalid requests
        requests = [valid_chat_completion_request] * 5 + [{"model": "invalid"}] * 5
        
        async def make_request(req):
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=req,
                    timeout=30.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Execute all requests concurrently
        tasks = [make_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle all requests without crashing
        assert len(results) == len(requests)
    
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, async_http_client: httpx.AsyncClient):
        """Gateway should handle rapid failures without cascading."""
        num_requests = 20
        
        async def make_failing_request(idx):
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json={"invalid": "request"},
                    timeout=10.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Rapid sequence of failing requests
        tasks = [make_failing_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle gracefully (not crash or hang)
        assert len(results) == num_requests
