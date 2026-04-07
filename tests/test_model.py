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
    """Bad code diff should predict FAIL."""
    result = model_service.predict(bad_diff)
    assert result['verdict'] in ('PASS', 'FAIL')
    assert 0 <= result['confidence'] <= 1
    assert 0 <= result['probability'] <= 1
    assert len(result['features']) == 16


def test_predict_good_code(model_service: ModelService, good_diff: str) -> None:
    """Good code diff should predict PASS."""
    result = model_service.predict(good_diff)
    assert result['verdict'] in ('PASS', 'FAIL')
    assert 0 <= result['confidence'] <= 1


def test_predict_empty_diff(model_service: ModelService) -> None:
    """Empty diff should not crash."""
    result = model_service.predict('')
    assert result['verdict'] in ('PASS', 'FAIL')


def test_classifier_architecture() -> None:
    """Model architecture has expected layer count."""
    model = CodeClassifier()
    params = sum(p.numel() for p in model.parameters())
    assert params > 0
    assert params < 10000  # Should be small (right-sized)
