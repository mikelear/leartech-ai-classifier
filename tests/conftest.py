"""Shared test fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import ModelService


@pytest.fixture()
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture()
def model_service() -> ModelService:
    """Loaded model service."""
    return ModelService('models/code_classifier.pt')


@pytest.fixture()
def bad_diff() -> str:
    """A code diff that should FAIL (hardcoded secrets, eval)."""
    return (
        "+ const API_KEY = 'sk-proj-abc123def456';\n"
        "+ const parsed = eval('(' + input + ')');\n"
        "+ document.getElementById('output').innerHTML = parsed.html;\n"
    )


@pytest.fixture()
def good_diff() -> str:
    """A code diff that should PASS (proper Angular patterns)."""
    return (
        "+ import { HttpClient } from '@angular/common/http';\n"
        '+ constructor(private http: HttpClient) {}\n'
        "+ this.http.get('/api/settings').subscribe(data => {});\n"
    )
