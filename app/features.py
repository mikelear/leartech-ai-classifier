"""Feature extraction from code diffs — turns text into a tensor."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ============================================================
# V1 features (16 regex counts) — kept for backwards compatibility
# ============================================================

FEATURE_NAMES_V1: list[str] = [
    'eval_calls',
    'innerHTML',
    'secret_names',
    'secret_patterns',
    'imports',
    'constructor',
    'async_patterns',
    'angular_services',
    'lines_added',
    'lines_removed',
    'total_lines',
    'functions',
    'control_flow',
    'error_handling',
    'test_related',
    'code_debt',
]

# ============================================================
# V3 features (28 = 16 original + 6 danger + 6 quality)
# ============================================================

DANGER_NAMES: list[str] = [
    'exec_system_pickle',
    'dangerous_html',
    'hardcoded_assignment',
    'untyped_python_def',
    'env_access',
    'base64_decode',
]

QUALITY_NAMES: list[str] = [
    'type_annotations',
    'return_type',
    'structured_types',
    'test_assertions',
    'explicit_errors',
    'async_typed',
]

# ============================================================
# Pipeline signal features (6 — from infrastructure, not code)
# ============================================================

PIPELINE_NAMES: list[str] = [
    'services_affected',
    'touches_critical',
    'unexpected_edges',
    'coverage_gaps',
    'e2e_passed',
    'leartech_violations',
]

# Neutral defaults when pipeline signals aren't available
PIPELINE_DEFAULTS: list[float] = [1, 0, 0, 0, 1, 0]

# ============================================================
# Combined feature names (all versions)
# ============================================================

FEATURE_NAMES_V3: list[str] = FEATURE_NAMES_V1 + DANGER_NAMES + QUALITY_NAMES
FEATURE_NAMES: list[str] = FEATURE_NAMES_V3 + PIPELINE_NAMES

NUM_HANDCRAFTED: int = len(FEATURE_NAMES_V3) + len(PIPELINE_NAMES)  # 34
NUM_FEATURES: int = NUM_HANDCRAFTED  # Updated by load_artefacts if TF-IDF exists

# Boost factor for hand-crafted features vs TF-IDF
HANDCRAFTED_BOOST: float = 3.0

# Module-level artefacts (loaded once at startup)
_tfidf: TfidfVectorizer | None = None
_scaler: StandardScaler | None = None
_num_features: int = NUM_HANDCRAFTED


def _load_tfidf_from_json(path: Path) -> TfidfVectorizer | None:
    """Reconstruct TfidfVectorizer from JSON params (no pickle)."""
    if not path.exists():
        return None
    with open(path) as f:
        params = json.load(f)
    tfidf = TfidfVectorizer(
        analyzer=params.get('analyzer', 'char_wb'),
        ngram_range=tuple(params['ngram_range']),
        max_features=params.get('max_features'),
        sublinear_tf=params.get('sublinear_tf', True),
    )
    tfidf.vocabulary_ = params['vocabulary']
    tfidf.idf_ = np.array(params['idf'])
    # Reconstruct internal state needed for transform()
    tfidf._tfidf._idf_diag = None  # forces recompute from idf_
    return tfidf


def _load_scaler_from_json(path: Path) -> StandardScaler | None:
    """Reconstruct StandardScaler from JSON params (no pickle)."""
    if not path.exists():
        return None
    with open(path) as f:
        params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(params['mean'])
    scaler.scale_ = np.array(params['scale'])
    scaler.var_ = np.array(params['var'])
    scaler.n_features_in_ = params['n_features_in']
    scaler.n_samples_seen_ = params.get('n_samples_seen', 1)
    return scaler


def load_artefacts(models_dir: str = 'models') -> None:
    """Load TF-IDF vectorizer and scaler from the models directory.

    Called once at startup. If artefacts don't exist, the service
    falls back to hand-crafted features only (v1 behaviour).

    Artefacts are stored as JSON (not pickle) to avoid deserialization
    security concerns. The JSON files contain only the numerical params
    needed to reconstruct the sklearn objects.
    """
    global _tfidf, _scaler, _num_features, NUM_FEATURES  # noqa: PLW0603

    models_path = Path(models_dir)

    _tfidf = _load_tfidf_from_json(models_path / 'tfidf_char.json')
    _scaler = _load_scaler_from_json(models_path / 'scaler.json')

    if _tfidf is not None:
        _num_features = NUM_HANDCRAFTED + len(_tfidf.get_feature_names_out())
    else:
        _num_features = NUM_HANDCRAFTED

    NUM_FEATURES = _num_features


def extract_features(diff: str) -> torch.Tensor:
    """Extract features from a code diff.

    Returns the full feature vector (v3 code + pipeline defaults + TF-IDF
    if available). This is the main entry point for the prediction API.
    """
    return extract_all_features(diff, pipeline_signals=None)


def extract_features_v1(diff: str) -> list[float]:
    """V1: 16 regex counts (Session 10 original)."""
    return [
        len(re.findall(r'eval\s*\(', diff)),
        len(re.findall(r'innerHTML', diff)),
        len(re.findall(r'(API_KEY|SECRET|PASSWORD|TOKEN)', diff, re.IGNORECASE)),
        len(re.findall(r'(sk-|ghp_|password|secret)', diff, re.IGNORECASE)),
        len(re.findall(r'^[\+].*import\s+', diff, re.MULTILINE)),
        len(re.findall(r'constructor', diff)),
        len(re.findall(r'(subscribe|Observable|Promise)', diff)),
        len(re.findall(r'(HttpClient|DomSanitizer|Injectable)', diff)),
        len(re.findall(r'^\+', diff, re.MULTILINE)),
        len(re.findall(r'^-', diff, re.MULTILINE)),
        len(diff.split('\n')),
        len(re.findall(r'(function|func |def |=>)', diff)),
        len(re.findall(r'(if |else|switch|case)', diff)),
        len(re.findall(r'(try|catch|error|Error)', diff)),
        len(re.findall(r'(test|spec|Test|Spec)', diff)),
        len(re.findall(r'(TODO|FIXME|HACK|XXX)', diff)),
    ]


def extract_danger_signals(diff: str) -> list[float]:
    """6 danger signals — code execution, XSS, secrets, untyped defs."""
    return [
        len(re.findall(r'(exec\s*\(|os\.system|subprocess\.call|pickle\.load)', diff)),
        len(re.findall(r'(dangerouslySetInnerHTML|\.innerHTML\s*=)', diff)),
        len(re.findall(r'(API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*["\']', diff, re.IGNORECASE)),
        1.0 if re.search(r'def\s+\w+\([^)]*\)\s*:', diff)
        and not re.search(r'def\s+\w+\([^)]*:\s*\w', diff) else 0.0,
        len(re.findall(r'(\.env|os\.environ|process\.env)', diff)),
        len(re.findall(r'(base64\.decode|atob\(|btoa\()', diff)),
    ]


def extract_quality_signals(diff: str) -> list[float]:
    """6 quality signals — type hints, structured types, assertions."""
    return [
        len(re.findall(r':\s*(str|int|float|bool|list|dict|Optional|Any)\b', diff)),
        len(re.findall(r'->\s*\w+', diff)),
        len(re.findall(r'(BaseModel|dataclass|interface\s+\w+|type\s+\w+\s+struct)', diff)),
        len(re.findall(r'(assert\s|expect\(|\.should\(|\.toBe\()', diff)),
        len(re.findall(r'(fmt\.Errorf|errors\.New|raise\s+\w+Error)', diff)),
        len(re.findall(r'(async\s+def|async\s+function|Observable<)', diff)),
    ]


def extract_features_v3(diff: str) -> list[float]:
    """V3: 28 features (16 original + 6 danger + 6 quality)."""
    return extract_features_v1(diff) + extract_danger_signals(diff) + extract_quality_signals(diff)


def extract_all_features(
    diff: str,
    pipeline_signals: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Full feature vector: 28 code + 6 pipeline + 200 TF-IDF.

    Args:
        diff: Code diff text.
        pipeline_signals: Optional dict from the pipeline with keys:
            services_affected, touches_critical, unexpected_edges,
            coverage_gaps, e2e_passed, leartech_violations.
            Defaults to neutral values when not provided.

    Returns:
        Float tensor of shape [NUM_FEATURES].
    """
    # 28 code features
    code = extract_features_v3(diff)

    # 6 pipeline signals (default to neutral if not provided)
    if pipeline_signals:
        infra = [
            float(pipeline_signals.get('services_affected', PIPELINE_DEFAULTS[0])),
            float(pipeline_signals.get('touches_critical', PIPELINE_DEFAULTS[1])),
            float(pipeline_signals.get('unexpected_edges', PIPELINE_DEFAULTS[2])),
            float(pipeline_signals.get('coverage_gaps', PIPELINE_DEFAULTS[3])),
            float(pipeline_signals.get('e2e_passed', PIPELINE_DEFAULTS[4])),
            float(pipeline_signals.get('leartech_violations', PIPELINE_DEFAULTS[5])),
        ]
    else:
        infra = list(PIPELINE_DEFAULTS)

    # Combine hand-crafted features
    handcrafted = code + infra  # 34 features

    # Scale and boost
    if _scaler is not None:
        scaled = _scaler.transform([handcrafted])[0] * HANDCRAFTED_BOOST
    else:
        scaled = np.array(handcrafted, dtype=np.float32)

    # Add TF-IDF if available
    if _tfidf is not None:
        tfidf_feat = _tfidf.transform([diff]).toarray()[0]
        full = np.concatenate([scaled, tfidf_feat])
    else:
        full = scaled

    return torch.tensor(full, dtype=torch.float32)
