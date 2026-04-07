"""Feature extraction from code diffs — turns text into a tensor."""

import re

import torch

FEATURE_NAMES: list[str] = [
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

NUM_FEATURES: int = len(FEATURE_NAMES)


def extract_features(diff: str) -> torch.Tensor:
    """Extract 16 numerical features from a code diff.

    Args:
        diff: The code diff text (unified diff format).

    Returns:
        A float tensor of shape [16] with feature counts.
    """
    features = [
        # Security signals
        len(re.findall(r'eval\s*\(', diff)),
        len(re.findall(r'innerHTML', diff)),
        len(re.findall(r'(API_KEY|SECRET|PASSWORD|TOKEN)', diff, re.IGNORECASE)),
        len(re.findall(r'(sk-|ghp_|password|secret)', diff, re.IGNORECASE)),
        # Code quality signals
        len(re.findall(r'^[\+].*import\s+', diff, re.MULTILINE)),
        len(re.findall(r'constructor', diff)),
        len(re.findall(r'(subscribe|Observable|Promise)', diff)),
        len(re.findall(r'(HttpClient|DomSanitizer|Injectable)', diff)),
        # Diff metrics
        len(re.findall(r'^\+', diff, re.MULTILINE)),
        len(re.findall(r'^-', diff, re.MULTILINE)),
        len(diff.split('\n')),
        len(re.findall(r'(function|func |def |=>)', diff)),
        len(re.findall(r'(if |else|switch|case)', diff)),
        len(re.findall(r'(try|catch|error|Error)', diff)),
        len(re.findall(r'(test|spec|Test|Spec)', diff)),
        len(re.findall(r'(TODO|FIXME|HACK|XXX)', diff)),
    ]
    return torch.tensor(features, dtype=torch.float32)
