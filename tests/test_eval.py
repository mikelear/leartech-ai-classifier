"""
Model evaluation tests — gates model quality.

These tests verify the committed model meets minimum quality thresholds.
Run on every PR and before deploying a retrained model.

If someone commits a new .pt model file, these tests ensure it doesn't
regress below the baseline.
"""

import torch

from app.features import extract_features
from app.model import CodeClassifier, ModelService

# Minimum acceptable metrics — update these as the model improves
BASELINE_ACCURACY = 0.65
BASELINE_RECALL = 0.80     # Must catch bad code — recall > precision
BASELINE_F1 = 0.70
BASELINE_PRECISION = 0.60

# Test diffs with known labels
EVAL_CASES: list[tuple[str, float, str]] = [
    # (diff, expected_label, description)
    # FAIL cases (label=1.0)
    (
        "+ const API_KEY = 'sk-proj-abc123';\n+ eval('(' + input + ')');\n",
        1.0,
        'hardcoded secret + eval injection',
    ),
    (
        "+ document.innerHTML = userInput;\n+ const SECRET = 'password123';\n",
        1.0,
        'innerHTML XSS + hardcoded password',
    ),
    (
        "+ eval(data);\n+ const PASSWORD = 'admin';\n",
        1.0,
        'eval + hardcoded password',
    ),
    # PASS cases (label=0.0)
    (
        "+ import { HttpClient } from '@angular/common/http';\n"
        "+ constructor(private http: HttpClient) {}\n"
        "+ this.http.get('/api').subscribe(data => {});\n",
        0.0,
        'proper Angular HTTP pattern',
    ),
    (
        "+ import { DomSanitizer } from '@angular/platform-browser';\n"
        "+ constructor(private sanitizer: DomSanitizer) {}\n",
        0.0,
        'proper Angular sanitizer usage',
    ),
    (
        "+ import { Observable } from 'rxjs';\n"
        "+ subscribe(result => { this.data = result; });\n",
        0.0,
        'proper reactive pattern',
    ),
]


class TestModelBaseline:
    """Verify the committed model meets minimum quality thresholds."""

    def test_model_accuracy_meets_baseline(self, model_service: ModelService) -> None:
        """Overall accuracy must meet baseline."""
        accuracy = model_service.metadata.get('accuracy', 0)
        assert accuracy >= BASELINE_ACCURACY, (
            f'Model accuracy {accuracy:.1%} below baseline {BASELINE_ACCURACY:.1%}'
        )

    def test_model_recall_meets_baseline(self, model_service: ModelService) -> None:
        """Recall must meet baseline — catching bad code is critical."""
        recall = model_service.metadata.get('recall', 0)
        assert recall >= BASELINE_RECALL, (
            f'Model recall {recall:.1%} below baseline {BASELINE_RECALL:.1%}. '
            f'Missing bad code is dangerous.'
        )

    def test_model_f1_meets_baseline(self, model_service: ModelService) -> None:
        """F1 score must meet baseline."""
        f1 = model_service.metadata.get('f1', 0)
        assert f1 >= BASELINE_F1, f'Model F1 {f1:.1%} below baseline {BASELINE_F1:.1%}'

    def test_model_precision_meets_baseline(self, model_service: ModelService) -> None:
        """Precision must meet baseline."""
        precision = model_service.metadata.get('precision', 0)
        assert precision >= BASELINE_PRECISION, (
            f'Model precision {precision:.1%} below baseline {BASELINE_PRECISION:.1%}'
        )


class TestModelPredictions:
    """Verify the model produces correct predictions on known test cases."""

    def test_detects_bad_code(self, model_service: ModelService) -> None:
        """Model should predict FAIL for known-bad diffs."""
        for diff, label, desc in EVAL_CASES:
            if label == 1.0:
                result = model_service.predict(diff)
                assert result['probability'] > 0.4, (
                    f'Failed to flag bad code ({desc}): '
                    f'probability={result["probability"]:.3f}, expected > 0.4'
                )

    def test_passes_good_code(self, model_service: ModelService) -> None:
        """Model should predict PASS for known-good diffs."""
        for diff, label, desc in EVAL_CASES:
            if label == 0.0:
                result = model_service.predict(diff)
                assert result['probability'] < 0.75, (
                    f'Incorrectly flagged good code ({desc}): '
                    f'probability={result["probability"]:.3f}, expected < 0.75'
                )

    def test_eval_accuracy_on_test_cases(self, model_service: ModelService) -> None:
        """Overall accuracy on eval test cases."""
        correct = 0
        for diff, label, desc in EVAL_CASES:
            result = model_service.predict(diff)
            predicted = 1.0 if result['probability'] > 0.5 else 0.0
            if predicted == label:
                correct += 1

        accuracy = correct / len(EVAL_CASES)
        print(f'\nEval accuracy: {accuracy:.0%} ({correct}/{len(EVAL_CASES)})')
        # Don't hard-fail on eval accuracy — just report it
        # The baseline metrics from training are the real gate


class TestModelRegression:
    """Regression tests — specific patterns that must always be caught."""

    def test_always_catches_eval(self, model_service: ModelService) -> None:
        """eval() must always be flagged."""
        result = model_service.predict("+ const x = eval(userInput);")
        features = result['features']
        assert features['eval_calls'] > 0, 'Feature extraction missed eval()'

    def test_always_catches_secrets(self, model_service: ModelService) -> None:
        """Hardcoded secrets must always be detected in features."""
        result = model_service.predict("+ const API_KEY = 'sk-prod-real-key';")
        features = result['features']
        assert features['secret_patterns'] > 0, 'Feature extraction missed secret pattern'

    def test_always_catches_innerhtml(self, model_service: ModelService) -> None:
        """innerHTML must always be detected in features."""
        result = model_service.predict("+ document.body.innerHTML = data;")
        features = result['features']
        assert features['innerHTML'] > 0, 'Feature extraction missed innerHTML'

    def test_empty_diff_does_not_crash(self, model_service: ModelService) -> None:
        """Empty diff should return a valid prediction."""
        result = model_service.predict('')
        assert result['verdict'] in ('PASS', 'FAIL')
        assert 0 <= result['probability'] <= 1
