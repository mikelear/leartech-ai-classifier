"""Integration tests for API endpoints."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    """Health endpoint responds."""
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        assert data['parameters'] > 0


def test_predict_endpoint(bad_diff: str) -> None:
    """Predict endpoint returns valid response."""
    with TestClient(app) as client:
        response = client.post('/predict', json={'diff': bad_diff})
        assert response.status_code == 200
        data = response.json()
        assert data['verdict'] in ('PASS', 'FAIL')
        assert 0 <= data['confidence'] <= 1
        assert 0 <= data['probability'] <= 1
        assert 'features' in data
        assert len(data['features']) == 16


def test_predict_empty_diff() -> None:
    """Predict with empty diff doesn't crash."""
    with TestClient(app) as client:
        response = client.post('/predict', json={'diff': ''})
        assert response.status_code == 200


def test_predict_missing_body() -> None:
    """Predict without body returns 422."""
    with TestClient(app) as client:
        response = client.post('/predict')
        assert response.status_code == 422


def test_model_info_endpoint() -> None:
    """Model info endpoint returns metadata."""
    with TestClient(app) as client:
        response = client.get('/model/info')
        assert response.status_code == 200
        data = response.json()
        assert data['parameters'] > 0
        assert data['training_examples'] > 0
        assert len(data['feature_names']) == 16


def test_openapi_docs() -> None:
    """OpenAPI docs are accessible."""
    with TestClient(app) as client:
        response = client.get('/docs')
        assert response.status_code == 200
