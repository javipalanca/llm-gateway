"""
Test configuration and fixtures for LLM Gateway E2E tests.

Provides:
- FastAPI test client with proper configuration
- Mock/real Grafana/Prometheus setup
- Database connections
- Test data generators
- Cleanup fixtures
"""

import asyncio
import os
import pytest
import httpx
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Load test environment
load_dotenv(os.path.join(os.path.dirname(__file__), ".env.test"))


# ============================================================================
# Constants
# ============================================================================

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://localhost:9000")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:9001")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# Test API keys
VALID_API_KEYS = [
    "sk-test-valid-key-1-1234567890abcdef",
    "sk-test-valid-key-2-1234567890abcdef",
]

INVALID_API_KEYS = [
    "sk-test-invalid-key-wrong",
    "invalid-format-key",
    "sk-test-expired-key",
    "",
]

# Test models (configured in models.yaml)
TEST_MODELS = [
    "deepseek-r1",
    "qwen-2.5-7b-instruct",
    "llama-2-70b-chat",
]


# ============================================================================
# Fixtures: HTTP Clients
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def http_client() -> httpx.Client:
    """Synchronous HTTP client for making requests to the controller."""
    client = httpx.Client(base_url=CONTROLLER_URL, timeout=60.0)
    yield client
    client.close()


@pytest.fixture(scope="session")
async def async_http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Asynchronous HTTP client for making parallel requests."""
    async with httpx.AsyncClient(base_url=CONTROLLER_URL, timeout=60.0) as client:
        yield client


@pytest.fixture
def test_client() -> TestClient:
    """TestClient for FastAPI application."""
    # Import here to avoid circular imports
    from controller.app import app
    return TestClient(app)


# ============================================================================
# Fixtures: Test Data
# ============================================================================

@pytest.fixture
def valid_chat_completion_request():
    """Valid OpenAI-compatible chat completion request."""
    return {
        "model": "deepseek-r1",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }


@pytest.fixture
def valid_chat_completion_stream_request():
    """Valid OpenAI-compatible chat completion request with streaming."""
    return {
        "model": "qwen-2.5-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in 50 words."}
        ],
        "stream": True,
        "temperature": 0.5,
        "top_p": 0.95
    }


@pytest.fixture
def invalid_chat_completion_requests():
    """Various invalid chat completion requests for error testing."""
    return [
        # Missing model
        {
            "messages": [{"role": "user", "content": "Hello"}],
        },
        # Missing messages
        {
            "model": "deepseek-r1",
        },
        # Empty messages
        {
            "model": "deepseek-r1",
            "messages": [],
        },
        # Invalid model
        {
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        # Invalid message format
        {
            "model": "deepseek-r1",
            "messages": [{"role": "user"}],  # missing content
        },
        # Invalid temperature
        {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 2.5,  # out of range [0, 2]
        },
        # Invalid max_tokens
        {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": -1,  # must be positive
        },
    ]


# ============================================================================
# Fixtures: Authorization Headers
# ============================================================================

@pytest.fixture
def headers_with_valid_key():
    """HTTP headers with valid API key."""
    return {
        "Authorization": f"Bearer {VALID_API_KEYS[0]}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def headers_with_invalid_key():
    """HTTP headers with invalid API key."""
    return {
        "Authorization": "Bearer invalid-key-format",
        "Content-Type": "application/json",
    }


@pytest.fixture
def headers_without_auth():
    """HTTP headers without authorization."""
    return {
        "Content-Type": "application/json",
    }


@pytest.fixture
def all_api_key_variants():
    """All valid and invalid API key variants for parametrized testing."""
    return {
        "valid": VALID_API_KEYS,
        "invalid": INVALID_API_KEYS,
        "missing": [None],
    }


# ============================================================================
# Fixtures: Mock Services
# ============================================================================

@pytest.fixture
def mock_litellm_response():
    """Standard mock response from LiteLLM backend."""
    return {
        "id": "chatcmpl-8v9eKj7D8X9X9X9X9X9X",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "2 + 2 equals 4."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


@pytest.fixture
def mock_streaming_chunk():
    """Standard mock streaming chunk from LiteLLM."""
    return 'data: {"id":"chatcmpl-8v9eKj","object":"chat.completion.chunk","created":1234567890,"model":"deepseek-r1","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}\n\n'


# ============================================================================
# Fixtures: Prometheus & Grafana
# ============================================================================

@pytest.fixture(scope="session")
def prometheus_client() -> httpx.Client:
    """Client for querying Prometheus."""
    return httpx.Client(base_url=PROMETHEUS_URL, timeout=10.0)


@pytest.fixture(scope="session")
def grafana_client() -> httpx.Client:
    """Client for querying Grafana API."""
    grafana_user = os.getenv("GRAFANA_ADMIN_USER", "admin")
    grafana_pass = os.getenv("GRAFANA_ADMIN_PASSWORD")
    
    client = httpx.Client(
        base_url=GRAFANA_URL,
        timeout=10.0,
        auth=(grafana_user, grafana_pass)
    )
    return client


# ============================================================================
# Fixtures: Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_containers():
    """Cleanup any test containers after each test."""
    yield
    # Cleanup code here if needed
    # e.g., stop any containers started during test


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests."""
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    yield
    # Teardown code here if needed


# ============================================================================
# Helper Functions
# ============================================================================

def generate_api_key(prefix: str = "sk-test") -> str:
    """Generate a test API key."""
    import secrets
    return f"{prefix}-{secrets.token_hex(16)}"


def generate_request_id() -> str:
    """Generate a unique request ID."""
    import uuid
    return str(uuid.uuid4())
