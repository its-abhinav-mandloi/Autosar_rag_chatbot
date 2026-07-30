"""
Tests for Health Check & Heartbeat endpoints.

Uses FastAPI TestClient to test the API without needing Ollama running.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestRootEndpoint:
    """Test the root / and /api endpoints."""

    def test_root_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_root_has_chat_ui(self):
        response = client.get("/")
        assert "AUTOSAR" in response.text

    def test_api_has_group_info(self):
        response = client.get("/api")
        data = response.json()
        assert "Group 151" in data.get("group", "")

    def test_api_has_endpoints(self):
        response = client.get("/api")
        data = response.json()
        assert "endpoints" in data
        assert "health" in data["endpoints"]


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should always return 200 (even if degraded)."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unavailable")

    def test_health_has_services(self):
        response = client.get("/health")
        data = response.json()
        assert "services" in data
        assert "ollama_api" in data["services"]
        assert "embedding_model" in data["services"]
        assert "vector_store" in data["services"]

    def test_health_has_uptime(self):
        response = client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestMetricsEndpoint:
    """Test the /health/metrics endpoint."""

    def test_metrics_returns_200(self):
        response = client.get("/health/metrics")
        assert response.status_code == 200

    def test_metrics_has_sections(self):
        response = client.get("/health/metrics")
        data = response.json()
        assert "queries" in data
        assert "ingestion" in data
        assert "feedback" in data


class TestCorrelationIdMiddleware:
    """Test correlation ID injection."""

    def test_response_has_correlation_id(self):
        response = client.get("/")
        assert "X-Correlation-ID" in response.headers

    def test_custom_correlation_id_passed_through(self):
        response = client.get("/", headers={"X-Correlation-ID": "test-123"})
        assert response.headers["X-Correlation-ID"] == "test-123"

    def test_response_has_timing(self):
        response = client.get("/")
        assert "X-Response-Time-Ms" in response.headers
