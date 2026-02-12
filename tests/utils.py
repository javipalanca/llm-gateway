"""
Utilities for E2E testing.

Provides:
- Performance measurement helpers
- Request builders
- Response validators
- Metrics collectors
"""

import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    model: str
    endpoint: str
    start_time: float
    end_time: float
    status_code: int
    error: Optional[str] = None
    tokens_sent: int = 0
    tokens_received: int = 0
    success: bool = False
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def tokens_per_second(self) -> float:
        """Throughput in tokens per second."""
        if self.duration_ms == 0:
            return 0
        return self.tokens_received / (self.duration_ms / 1000)


@dataclass
class TestResults:
    """Results for a test run."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    request_metrics: List[RequestMetrics]
    start_time: datetime
    end_time: datetime
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_requests == 0:
            return 0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_response_time_ms(self) -> float:
        """Average response time in milliseconds."""
        if not self.request_metrics:
            return 0
        return sum(m.duration_ms for m in self.request_metrics) / len(self.request_metrics)
    
    @property
    def p50_response_time_ms(self) -> float:
        """50th percentile response time."""
        if not self.request_metrics:
            return 0
        sorted_times = sorted([m.duration_ms for m in self.request_metrics])
        return sorted_times[len(sorted_times) // 2]
    
    @property
    def p95_response_time_ms(self) -> float:
        """95th percentile response time."""
        if not self.request_metrics:
            return 0
        sorted_times = sorted([m.duration_ms for m in self.request_metrics])
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def p99_response_time_ms(self) -> float:
        """99th percentile response time."""
        if not self.request_metrics:
            return 0
        sorted_times = sorted([m.duration_ms for m in self.request_metrics])
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def max_response_time_ms(self) -> float:
        """Maximum response time."""
        if not self.request_metrics:
            return 0
        return max(m.duration_ms for m in self.request_metrics)
    
    @property
    def min_response_time_ms(self) -> float:
        """Minimum response time."""
        if not self.request_metrics:
            return 0
        return min(m.duration_ms for m in self.request_metrics)


class RequestBuilder:
    """Builder for creating test requests."""
    
    @staticmethod
    def chat_completion(
        model: str,
        message: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build a chat completion request."""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": message})
        
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        return request
    
    @staticmethod
    def multi_turn_conversation(
        model: str,
        turns: List[tuple],  # List of (role, content) tuples
        **kwargs
    ) -> Dict[str, Any]:
        """Build a multi-turn conversation request."""
        messages = [
            {"role": role, "content": content}
            for role, content in turns
        ]
        
        return {
            "model": model,
            "messages": messages,
            **kwargs
        }


class ResponseValidator:
    """Validator for API responses."""
    
    @staticmethod
    def validate_chat_completion(response: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate chat completion response format."""
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        
        for field in required_fields:
            if field not in response:
                return False, f"Missing required field: {field}"
        
        if not isinstance(response["choices"], list) or len(response["choices"]) == 0:
            return False, "choices must be a non-empty list"
        
        choice = response["choices"][0]
        if "message" not in choice or "content" not in choice["message"]:
            return False, "Missing message content in choice"
        
        return True, None
    
    @staticmethod
    def validate_streaming_chunk(chunk: str) -> tuple[bool, Optional[str]]:
        """Validate streaming response chunk."""
        if not chunk.startswith("data: "):
            return False, "Chunk should start with 'data: '"
        
        try:
            json.loads(chunk[6:])
            return True, None
        except json.JSONDecodeError:
            return False, "Chunk contains invalid JSON"
    
    @staticmethod
    def validate_error_response(response: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate error response format."""
        if "error" not in response:
            return False, "Error response missing 'error' field"
        
        error = response["error"]
        if not isinstance(error, dict):
            return False, "error field should be a dict"
        
        if "message" not in error:
            return False, "error should contain 'message'"
        
        return True, None


class PerformanceAnalyzer:
    """Analyze performance metrics."""
    
    @staticmethod
    def analyze(metrics: List[RequestMetrics]) -> Dict[str, Any]:
        """Analyze request metrics."""
        if not metrics:
            return {}
        
        durations = [m.duration_ms for m in metrics]
        tokens_per_sec = [m.tokens_per_second for m in metrics if m.tokens_per_second > 0]
        
        return {
            "total_requests": len(metrics),
            "successful": sum(1 for m in metrics if m.success),
            "failed": sum(1 for m in metrics if not m.success),
            "response_time": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "p50": sorted(durations)[len(durations) // 2],
                "p95": sorted(durations)[int(len(durations) * 0.95)],
                "p99": sorted(durations)[int(len(durations) * 0.99)],
            },
            "throughput": {
                "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
                "min_tokens_per_sec": min(tokens_per_sec) if tokens_per_sec else 0,
                "max_tokens_per_sec": max(tokens_per_sec) if tokens_per_sec else 0,
            }
        }


class MetricsCollector:
    """Collect metrics from test runs."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.metrics: List[RequestMetrics] = []
        self.start_time = datetime.now()
    
    def record_request(
        self,
        request_id: str,
        model: str,
        endpoint: str,
        duration_sec: float,
        status_code: int,
        success: bool = True,
        error: Optional[str] = None,
        tokens_sent: int = 0,
        tokens_received: int = 0,
    ) -> None:
        """Record a single request."""
        metric = RequestMetrics(
            request_id=request_id,
            model=model,
            endpoint=endpoint,
            start_time=time.time() - duration_sec,
            end_time=time.time(),
            status_code=status_code,
            error=error,
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
            success=success,
        )
        self.metrics.append(metric)
    
    def get_results(self) -> TestResults:
        """Get final test results."""
        successful = sum(1 for m in self.metrics if m.success)
        failed = sum(1 for m in self.metrics if not m.success)
        
        return TestResults(
            test_name=self.test_name,
            total_requests=len(self.metrics),
            successful_requests=successful,
            failed_requests=failed,
            request_metrics=self.metrics,
            start_time=self.start_time,
            end_time=datetime.now(),
        )
    
    def print_summary(self) -> None:
        """Print summary of test results."""
        results = self.get_results()
        duration = (results.end_time - results.start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"Test: {results.test_name}")
        print(f"{'='*70}")
        print(f"Total Requests:      {results.total_requests}")
        print(f"Successful:          {results.successful_requests}")
        print(f"Failed:              {results.failed_requests}")
        print(f"Success Rate:        {results.success_rate:.1f}%")
        print(f"Duration:            {duration:.2f}s")
        print(f"RPS:                 {results.total_requests / duration:.2f}")
        print()
        print(f"Response Time (ms):")
        print(f"  Min:               {results.min_response_time_ms:.2f}")
        print(f"  Avg:               {results.avg_response_time_ms:.2f}")
        print(f"  P50:               {results.p50_response_time_ms:.2f}")
        print(f"  P95:               {results.p95_response_time_ms:.2f}")
        print(f"  P99:               {results.p99_response_time_ms:.2f}")
        print(f"  Max:               {results.max_response_time_ms:.2f}")
        print(f"{'='*70}\n")


def assert_valid_response(response: Dict[str, Any], expect_success: bool = True) -> None:
    """Assert that a response is valid."""
    if expect_success:
        assert "choices" in response or "error" not in response
    else:
        assert "error" in response or response.get("status_code", 200) >= 400
