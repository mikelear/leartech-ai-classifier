"""Tests for feature extraction."""

import torch

from app.features import (
    FEATURE_NAMES_V1,
    FEATURE_NAMES_V3,
    PIPELINE_NAMES,
    extract_all_features,
    extract_danger_signals,
    extract_features,
    extract_features_v1,
    extract_features_v3,
    extract_quality_signals,
)


def test_v1_feature_count() -> None:
    """V1 extractor returns 16 features."""
    features = extract_features_v1('+ some code')
    assert len(features) == 16
    assert len(features) == len(FEATURE_NAMES_V1)


def test_v3_feature_count() -> None:
    """V3 extractor returns 28 features (16 + 6 danger + 6 quality)."""
    features = extract_features_v3('+ some code')
    assert len(features) == 28
    assert len(features) == len(FEATURE_NAMES_V3)


def test_danger_signals() -> None:
    """Danger signals detect eval, exec, pickle, secrets."""
    danger = extract_danger_signals('+ result = eval(data)\n+ exec(cmd)\n+ pickle.load(f)')
    assert danger[0] > 0  # exec_system_pickle (exec + pickle.load)
    assert danger[3] == 0.0  # untyped_python_def — not present


def test_quality_signals() -> None:
    """Quality signals detect type hints, structured types."""
    quality = extract_quality_signals(
        '+ def create(name: str, age: int) -> dict:\n'
        '+ class Item(BaseModel):\n'
        '+     assert result == expected',
    )
    assert quality[0] > 0  # type_annotations (str, int)
    assert quality[1] > 0  # return_type (-> dict)
    assert quality[2] > 0  # structured_types (BaseModel)
    assert quality[3] > 0  # test_assertions (assert)


def test_extract_features_returns_tensor() -> None:
    """Main extract_features returns a float32 tensor."""
    features = extract_features('+ some code')
    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32


def test_extract_all_features_without_signals() -> None:
    """extract_all_features works without pipeline signals."""
    features = extract_all_features('+ some code', pipeline_signals=None)
    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32


def test_extract_all_features_with_signals() -> None:
    """extract_all_features works with pipeline signals."""
    signals = {
        'services_affected': 3,
        'touches_critical': 1.0,
        'unexpected_edges': 1,
        'coverage_gaps': 2,
        'e2e_passed': 0.0,
        'leartech_violations': 2,
    }
    features = extract_all_features('+ some code', pipeline_signals=signals)
    assert isinstance(features, torch.Tensor)


def test_eval_detection(bad_diff: str) -> None:
    """Detects eval() calls in v1 features."""
    features = extract_features_v1(bad_diff)
    assert features[0] > 0  # eval_calls


def test_innerhtml_detection(bad_diff: str) -> None:
    """Detects innerHTML usage in v1 features."""
    features = extract_features_v1(bad_diff)
    assert features[1] > 0  # innerHTML


def test_secret_detection(bad_diff: str) -> None:
    """Detects hardcoded secrets in v1 features."""
    features = extract_features_v1(bad_diff)
    assert features[2] > 0  # secret_names


def test_import_detection(good_diff: str) -> None:
    """Detects import statements."""
    features = extract_features_v1(good_diff)
    assert features[4] > 0  # imports


def test_empty_diff() -> None:
    """Empty diff has no security or quality signals."""
    features = extract_features_v1('')
    assert sum(features[0:8]) == 0
    assert features[10] == 1  # total_lines = 1 (empty splits to [''])


def test_pipeline_feature_names() -> None:
    """Pipeline feature names are defined."""
    assert len(PIPELINE_NAMES) == 6
    assert 'e2e_passed' in PIPELINE_NAMES
    assert 'leartech_violations' in PIPELINE_NAMES
