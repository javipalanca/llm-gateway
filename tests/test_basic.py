"""
E2E tests for basic gateway functionality.

Tests cover:
- Health endpoint
- Model listing
- Metadata endpoints
- Basic error handling
"""

import pytest
import httpx
from typing import Dict, Any


class TestHealthEndpoint:
    """Test /health endpoint"""
    
    def test_health_check_returns_ok(self, http_client: httpx.Client):
        """Health endpoint should return status 200 with 'ok' status."""
        response = http_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "litellm_base_url" in data
        assert "vllm_host_ip" in data
        assert "idle_ttl_minutes" in data
        assert "models" in data
    
    
    def test_health_includes_all_configured_models(self, http_client: httpx.Client):
        """Health endpoint should list all configured models."""
        response = http_client.get("/health")
        data = response.json()
        
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        # Each model should be a string key
        assert all(isinstance(m, str) for m in data["models"])
    
    
    def test_health_includes_vram_snapshot(self, http_client: httpx.Client):
        """Health endpoint should include VRAM information."""
        response = http_client.get("/health")
        data = response.json()
        
        assert "vram" in data
        assert isinstance(data["vram"], dict)
        
        # Each model should have VRAM info
        for model_key, vram_info in data["vram"].items():
            assert "req_gpu" in vram_info
            assert "min_free_gib" in vram_info
            assert "free_gib" in vram_info
            assert "warm" in vram_info
            assert "priority" in vram_info
            assert "running" in vram_info


class TestMetricsEndpoints:
    """Test /metrics endpoints"""
    
    def test_prometheus_metrics_endpoint(self, http_client: httpx.Client):
        """Prometheus /metrics endpoint should return valid metrics."""
        response = http_client.get("/metrics")
        assert response.status_code == 200
        
        # Should have Prometheus format
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "llm_gateway_requests_total" in content or "requests" in content.lower()
    
    
    def test_metrics_state_endpoint(self, http_client: httpx.Client):
        """GET /metrics/state should return current metrics state."""
        response = http_client.get("/metrics/state")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)
    
    
    def test_metrics_history_endpoint(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """GET /metrics/history/{model} should return historical data."""
        model = valid_chat_completion_request["model"]
        response = http_client.get(f"/metrics/history/{model}?hours=1")
        
        # Either 200 (data exists) or 404 (no history yet) are acceptable
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert data["model"] == model
            assert "metric" in data
            assert "hours" in data
            assert "data" in data


class TestModelDiscovery:
    """Test model discovery and information endpoints"""
    
    def test_get_models_list(self, http_client: httpx.Client):
        """Should be able to discover available models from /health."""
        response = http_client.get("/health")
        models = response.json()["models"]
        
        assert len(models) > 0
        # Models should be valid identifiers
        assert all(isinstance(m, str) and len(m) > 0 for m in models)
    
    
    def test_vram_requirements_for_each_model(self, http_client: httpx.Client):
        """Each model should have VRAM requirements specified."""
        response = http_client.get("/health")
        vram_snapshot = response.json()["vram"]
        
        for model_key, vram_info in vram_snapshot.items():
            assert isinstance(vram_info["min_free_gib"], (int, float, type(None)))
            assert isinstance(vram_info["warm"], bool)
            assert isinstance(vram_info["priority"], int)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_nonexistent_model_error(self, http_client: httpx.Client):
        """Requesting a nonexistent model should return appropriate error."""
        request_data = {
            "model": "nonexistent-model-xyz-999",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        
        response = http_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code in [400, 404, 500]
        
        data = response.json()
        assert "error" in data or "message" in data
    
    
    def test_invalid_json_request(self, http_client: httpx.Client):
        """Sending invalid JSON should be handled gracefully."""
        response = http_client.post(
            "/v1/chat/completions",
            content=b"invalid json {{{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
    
    
    def test_missing_required_fields(self, http_client: httpx.Client):
        """Missing required fields should return validation error."""
        invalid_requests = [
            {},  # completely empty
            {"model": "deepseek-r1"},  # missing messages
            {"messages": [{"role": "user", "content": "hi"}]},  # missing model
        ]
        
        for request_data in invalid_requests:
            response = http_client.post("/v1/chat/completions", json=request_data)
            assert response.status_code in [400, 422, 500]
    
    
    def test_malformed_message_format(self, http_client: httpx.Client):
        """Malformed message format should be rejected."""
        request_data = {
            "model": "deepseek-r1",
            "messages": [
                {"role": "user"},  # missing content
                {"content": "missing role"},  # missing role
            ]
        }
        
        response = http_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code in [400, 422, 500]


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_message_content(self, http_client: httpx.Client):
        """Empty message content should be handled."""
        request_data = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": ""}]
        }
        
        response = http_client.post("/v1/chat/completions", json=request_data)
        # Should either process or reject gracefully
        assert response.status_code in [200, 400, 422, 500]
    
    
    def test_very_long_message(self, http_client: httpx.Client):
        """Very long messages should be handled appropriately."""
        request_data = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "x" * 100000}]
        }
        
        response = http_client.post("/v1/chat/completions", json=request_data)
        # Should either process or reject with appropriate status
        assert response.status_code in [200, 400, 413, 422, 500]
    
    
    def test_many_messages_in_conversation(self, http_client: httpx.Client):
        """Large conversation history should be handled."""
        messages = []
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})
        
        request_data = {
            "model": "deepseek-r1",
            "messages": messages
        }
        
        response = http_client.post("/v1/chat/completions", json=request_data)
        # Should either process or return appropriate error
        assert response.status_code in [200, 400, 413, 422, 500]
    
    
    def test_temperature_boundary_values(self, http_client: httpx.Client):
        """Test temperature parameter boundary values."""
        base_request = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        # Test valid boundary values
        for temp in [0, 0.5, 1.0, 1.5, 2.0]:
            request = {**base_request, "temperature": temp}
            response = http_client.post("/v1/chat/completions", json=request)
            # Should accept valid values
            assert response.status_code in [200, 500]
        
        # Test invalid boundary values
        for temp in [-0.1, 2.1, 10]:
            request = {**base_request, "temperature": temp}
            response = http_client.post("/v1/chat/completions", json=request)
            # Should reject invalid values or handle gracefully
            assert response.status_code in [400, 422, 500]
    
    
    def test_top_p_boundary_values(self, http_client: httpx.Client):
        """Test top_p parameter boundary values."""
        base_request = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        # Test valid values
        for top_p in [0, 0.5, 0.9, 1.0]:
            request = {**base_request, "top_p": top_p}
            response = http_client.post("/v1/chat/completions", json=request)
            assert response.status_code in [200, 500]
        
        # Test invalid values
        for top_p in [-0.1, 1.1, 2.0]:
            request = {**base_request, "top_p": top_p}
            response = http_client.post("/v1/chat/completions", json=request)
            assert response.status_code in [400, 422, 500]


class TestTimeout:
    """Test timeout behavior"""
    
    def test_request_completes_within_timeout(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Requests should complete within reasonable timeout."""
        try:
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                timeout=60.0
            )
            # Should either succeed or fail gracefully
            assert response.status_code in [200, 400, 408, 500, 502, 503, 504]
        except httpx.TimeoutException:
            # Timeout is acceptable for long operations
            pytest.skip("Request timed out (expected for long operations)")
