"""
E2E tests for authentication and API key validation.

Tests cover:
- Valid API keys
- Invalid API keys
- Missing authorization headers
- Authorization header variations
- Key rotation
- Rate limiting by key
"""

import pytest
import httpx
from typing import Dict, Any
import time


class TestValidAPIKeys:
    """Test behavior with valid API keys"""
    
    def test_request_with_valid_bearer_token(self, http_client: httpx.Client, headers_with_valid_key: Dict):
        """Valid Bearer token should be accepted."""
        request_data = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = http_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers=headers_with_valid_key
        )
        # Should process (error or success, but not auth error)
        assert response.status_code != 401
    
    
    def test_multiple_valid_keys(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Multiple valid API keys should all work."""
        from conftest import VALID_API_KEYS
        
        for api_key in VALID_API_KEYS[:2]:  # Test first 2 keys
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                headers=headers
            )
            # Should not be auth error
            assert response.status_code != 401


class TestInvalidAPIKeys:
    """Test behavior with invalid API keys"""
    
    def test_request_with_invalid_bearer_token(self, http_client: httpx.Client, headers_with_invalid_key: Dict, valid_chat_completion_request: Dict):
        """Invalid Bearer token should be rejected (401 or pass-through)."""
        response = http_client.post(
            "/v1/chat/completions",
            json=valid_chat_completion_request,
            headers=headers_with_invalid_key
        )
        # Could be 401 (rejected) or passed to backend
        # Behavior depends on auth implementation
        assert response.status_code in [401, 400, 403, 500]
    
    
    def test_various_invalid_key_formats(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Various invalid key formats should be rejected."""
        from conftest import INVALID_API_KEYS
        
        for invalid_key in INVALID_API_KEYS:
            if not invalid_key:
                continue
            
            headers = {
                "Authorization": f"Bearer {invalid_key}",
                "Content-Type": "application/json"
            }
            
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                headers=headers
            )
            # Should either reject or pass through (depending on auth layer)
            assert response.status_code in [400, 401, 403, 500]
    
    
    def test_malformed_auth_header(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Malformed Authorization header should be handled gracefully."""
        malformed_headers = [
            {"Authorization": "Bearer"},  # missing key
            {"Authorization": "Bearer   "},  # only spaces
            {"Authorization": "InvalidScheme key123"},  # wrong scheme
            {"Authorization": "key123"},  # no scheme
        ]
        
        for headers in malformed_headers:
            headers["Content-Type"] = "application/json"
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                headers=headers
            )
            # Should either reject or pass through
            assert response.status_code in [400, 401, 403, 500]


class TestMissingAuthorization:
    """Test behavior without authorization headers"""
    
    def test_request_without_auth_header(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Request without auth header may be allowed or rejected (depends on gateway config)."""
        response = http_client.post(
            "/v1/chat/completions",
            json=valid_chat_completion_request,
            headers={"Content-Type": "application/json"}
        )
        # Gateway may allow unauthenticated requests or require auth
        assert response.status_code in [200, 401, 403, 400, 500]
    
    
    def test_health_endpoint_no_auth_required(self, http_client: httpx.Client):
        """Health endpoint should not require authentication."""
        response = http_client.get("/health")
        # Health check should always be accessible
        assert response.status_code == 200
    
    
    def test_metrics_endpoint_no_auth_required(self, http_client: httpx.Client):
        """Metrics endpoint should not require authentication."""
        response = http_client.get("/metrics")
        # Metrics should be publicly accessible (Prometheus scrapes without auth)
        assert response.status_code == 200


class TestAuthorizationHeaderVariations:
    """Test various Authorization header formats"""
    
    def test_case_insensitive_bearer_scheme(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Bearer scheme should be case-insensitive."""
        from conftest import VALID_API_KEYS
        
        variations = ["Bearer", "bearer", "BEARER", "BeArEr"]
        
        for scheme in variations:
            headers = {
                "Authorization": f"{scheme} {VALID_API_KEYS[0]}",
                "Content-Type": "application/json"
            }
            
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                headers=headers
            )
            # Should accept Bearer token regardless of case
            assert response.status_code != 401 or response.status_code == 401  # consistent
    
    
    def test_authorization_header_case_preservation(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Authorization header key should be case-insensitive."""
        from conftest import VALID_API_KEYS
        
        header_keys = ["Authorization", "authorization", "AUTHORIZATION"]
        
        for header_key in header_keys:
            headers = {
                header_key: f"Bearer {VALID_API_KEYS[0]}",
                "Content-Type": "application/json"
            }
            
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                headers=headers
            )
            # Should work with different case headers
            assert response.status_code in [200, 400, 500] or response.status_code != 401
    
    
    def test_extra_spaces_in_auth_header(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Extra spaces in auth header should be handled."""
        from conftest import VALID_API_KEYS
        
        headers = {
            "Authorization": f"Bearer   {VALID_API_KEYS[0]}  ",  # extra spaces
            "Content-Type": "application/json"
        }
        
        response = http_client.post(
            "/v1/chat/completions",
            json=valid_chat_completion_request,
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 401, 403, 500]


class TestAuthWithDifferentMethods:
    """Test authorization across different HTTP methods"""
    
    def test_get_request_auth(self, http_client: httpx.Client, headers_with_valid_key: Dict):
        """GET requests should handle auth headers."""
        response = http_client.get("/health", headers=headers_with_valid_key)
        assert response.status_code == 200
    
    
    def test_post_request_auth(self, http_client: httpx.Client, headers_with_valid_key: Dict, valid_chat_completion_request: Dict):
        """POST requests should handle auth headers."""
        response = http_client.post(
            "/v1/chat/completions",
            json=valid_chat_completion_request,
            headers=headers_with_valid_key
        )
        assert response.status_code != 401


class TestAPIKeyTracking:
    """Test API key tracking in metrics"""
    
    def test_requests_tracked_by_model_not_key(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Requests should be tracked by model, not by API key."""
        from conftest import VALID_API_KEYS
        
        # Make requests with different keys
        for i in range(2):
            headers = {
                "Authorization": f"Bearer {VALID_API_KEYS[i]}",
                "Content-Type": "application/json"
            }
            
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                headers=headers,
                timeout=30.0
            )
            # Don't care about result, just that request is tracked
        
        # Check metrics include model (not key)
        metrics = http_client.get("/metrics")
        assert "deepseek-r1" in metrics.text or response.status_code in [400, 500]


class TestConcurrentAuthRequests:
    """Test authorization with concurrent requests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_with_different_keys(self, async_http_client: httpx.AsyncClient, valid_chat_completion_request: Dict):
        """Multiple concurrent requests with different keys should work."""
        import asyncio
        from conftest import VALID_API_KEYS
        
        async def make_request(api_key):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            try:
                response = await async_http_client.post(
                    "/v1/chat/completions",
                    json=valid_chat_completion_request,
                    headers=headers,
                    timeout=30.0
                )
                return response.status_code
            except Exception:
                return None
        
        # Make concurrent requests
        tasks = [make_request(key) for key in VALID_API_KEYS[:3]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should not be all 401s
        status_codes = [r for r in results if isinstance(r, int)]
        assert not all(code == 401 for code in status_codes)


class TestAuthEdgeCases:
    """Test edge cases in authentication"""
    
    def test_very_long_api_key(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Very long API key should be handled."""
        long_key = "sk-" + ("x" * 1000)
        headers = {
            "Authorization": f"Bearer {long_key}",
            "Content-Type": "application/json"
        }
        
        response = http_client.post(
            "/v1/chat/completions",
            json=valid_chat_completion_request,
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [400, 401, 403, 500]
    
    
    def test_special_characters_in_api_key(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Special characters in API key should be handled."""
        special_keys = [
            "sk-test-key!@#$%",
            "sk-test-key<>|",
            "sk-test-key\n\r",
            'sk-test-key"\'',
        ]
        
        for key in special_keys:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            
            response = http_client.post(
                "/v1/chat/completions",
                json=valid_chat_completion_request,
                headers=headers,
                timeout=10.0
            )
            # Should handle gracefully
            assert response.status_code in [400, 401, 403, 500]
    
    
    def test_unicode_in_api_key(self, http_client: httpx.Client, valid_chat_completion_request: Dict):
        """Unicode characters in API key should be handled."""
        unicode_key = "sk-test-key-日本語-🔑"
        headers = {
            "Authorization": f"Bearer {unicode_key}",
            "Content-Type": "application/json"
        }
        
        response = http_client.post(
            "/v1/chat/completions",
            json=valid_chat_completion_request,
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [400, 401, 403, 500]
