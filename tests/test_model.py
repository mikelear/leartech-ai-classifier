"""Tests for model loading and prediction."""

from app.model import CodeClassifier, ModelService


def test_model_loads() -> None:
    """Model loads from .pt file without errors."""
    service = ModelService('models/code_classifier.pt')
    assert service.model is not None


def test_model_metadata() -> None:
    """Model metadata is populated."""
    service = ModelService('models/code_classifier.pt')
    info = service.info()
    assert info['parameters'] > 0
    assert info['training_examples'] > 0
    assert 0 <= info['accuracy'] <= 1


def test_predict_bad_code(model_service: ModelService, bad_diff: str) -> None:
    """Bad code diff prediction returns valid structure."""
    result = model_service.predict(bad_diff)
    assert result['verdict'] in ('PASS', 'FAIL')
    assert 0 <= result['confidence'] <= 1
    assert 0 <= result['probability'] <= 1
    assert len(result['features']) > 0


def test_predict_good_code(model_service: ModelService, good_diff: str) -> None:
    """Good code diff prediction returns valid structure."""
    result = model_service.predict(good_diff)
    assert result['verdict'] in ('PASS', 'FAIL')
    assert 0 <= result['confidence'] <= 1


def test_predict_with_pipeline_signals(model_service: ModelService, bad_diff: str) -> None:
    """Prediction works with pipeline signals."""
    signals = {
        'services_affected': 5,
        'touches_critical': 1.0,
        'unexpected_edges': 2,
        'coverage_gaps': 3,
        'e2e_passed': 0.0,
        'leartech_violations': 2,
    }
    result = model_service.predict(bad_diff, pipeline_signals=signals)
    assert result['verdict'] in ('PASS', 'FAIL')


def test_predict_empty_diff(model_service: ModelService) -> None:
    """Empty diff should not crash."""
    result = model_service.predict('')
    assert result['verdict'] in ('PASS', 'FAIL')


def test_classifier_architecture() -> None:
    """Model architecture adapts to input dimension."""
    small = CodeClassifier(input_dim=16)
    large = CodeClassifier(input_dim=234)
    small_params = sum(p.numel() for p in small.parameters())
    large_params = sum(p.numel() for p in large.parameters())
    assert small_params < large_params
    assert small_params > 0
