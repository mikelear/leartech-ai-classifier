"""Tests for feature extraction."""

import torch

from app.features import NUM_FEATURES, extract_features


def test_feature_count() -> None:
    """Extract features returns correct number of features."""
    features = extract_features('+ some code')
    assert features.shape == (NUM_FEATURES,)


def test_feature_dtype() -> None:
    """Features are float32 tensors."""
    features = extract_features('+ some code')
    assert features.dtype == torch.float32


def test_eval_detection(bad_diff: str) -> None:
    """Detects eval() calls."""
    features = extract_features(bad_diff)
    assert features[0].item() > 0  # eval_calls


def test_innerhtml_detection(bad_diff: str) -> None:
    """Detects innerHTML usage."""
    features = extract_features(bad_diff)
    assert features[1].item() > 0  # innerHTML


def test_secret_detection(bad_diff: str) -> None:
    """Detects hardcoded secrets."""
    features = extract_features(bad_diff)
    assert features[2].item() > 0  # secret_names


def test_import_detection(good_diff: str) -> None:
    """Detects import statements."""
    features = extract_features(good_diff)
    assert features[4].item() > 0  # imports


def test_constructor_detection(good_diff: str) -> None:
    """Detects constructor patterns."""
    features = extract_features(good_diff)
    assert features[5].item() > 0  # constructor


def test_empty_diff() -> None:
    """Empty diff has no security or quality signals."""
    features = extract_features('')
    # Security + quality features (0-7) should all be zero
    assert features[0:8].sum().item() == 0
    # total_lines is 1 (empty string splits to [''])
    assert features[10].item() == 1


def test_lines_added() -> None:
    """Counts added lines."""
    diff = '+ line1\n+ line2\n+ line3'
    features = extract_features(diff)
    assert features[8].item() == 3  # lines_added
