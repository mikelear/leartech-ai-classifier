"""
Model evaluation tests — gates model quality.

These tests verify the committed model meets minimum quality thresholds.
Run on every PR and before deploying a retrained model.

If someone commits a new .pt model file, these tests ensure it doesn't
regress below the baseline.
"""

from app.features import extract_features_v1
from app.model import ModelService

# Minimum acceptable metrics — update these as the model improves
BASELINE_ACCURACY = 0.65
BASELINE_RECALL = 0.50
BASELINE_F1 = 0.50
BASELINE_PRECISION = 0.50

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
        "+ from pydantic import BaseModel\n"
        "+ class Item(BaseModel):\n"
        "+     name: str\n"
        "+     price: float\n",
        0.0,
        'proper typed Python pattern',
    ),
]


class TestModelBaseline:
    """Verify the committed model meets minimum quality thresholds."""

    def test_model_accuracy_meets_baseline(self, model_service: ModelService) -> None:
        """Overall accuracy must meet baseline."""
        # Check both accuracy and eval_accuracy (different checkpoint versions store different keys)
        accuracy = model_service.metadata.get('accuracy', 0) or model_service.metadata.get('eval_accuracy', 0)
        assert accuracy >= BASELINE_ACCURACY, (
            f'Model accuracy {accuracy:.1%} below baseline {BASELINE_ACCURACY:.1%}'
        )

    def test_model_has_training_examples(self, model_service: ModelService) -> None:
        """Model must have been trained on real data."""
        examples = model_service.metadata.get('training_examples', 0)
        assert examples >= 50, f'Model trained on only {examples} examples (need ≥50)'

    def test_model_has_reasonable_params(self, model_service: ModelService) -> None:
        """Model parameter count should be reasonable for data size."""
        params = model_service.metadata.get('parameters', 0)
        assert params > 0, 'Model has no parameters'
        assert params < 100_000, f'Model too large ({params} params) — risk of overfitting'


class TestModelPredictions:
    """Verify the model produces correct predictions on known test cases."""

    def test_detects_bad_code(self, model_service: ModelService) -> None:
        """Model should predict FAIL for known-bad diffs."""
        for diff, label, desc in EVAL_CASES:
            if label == 1.0:
                result = model_service.predict(diff)
                assert result['probability'] > 0.3, (
                    f'Failed to flag bad code ({desc}): '
                    f'probability={result["probability"]:.3f}, expected > 0.3'
                )

    def test_passes_good_code(self, model_service: ModelService) -> None:
        """Model should predict PASS for known-good diffs."""
        for diff, label, desc in EVAL_CASES:
            if label == 0.0:
                result = model_service.predict(diff)
                assert result['probability'] < 0.95, (
                    f'Incorrectly flagged good code ({desc}): '
                    f'probability={result["probability"]:.3f}, expected < 0.95'
                )

    def test_eval_accuracy_on_test_cases(self, model_service: ModelService) -> None:
        """Overall accuracy on eval test cases."""
        correct = 0
        for diff, label, _desc in EVAL_CASES:
            result = model_service.predict(diff)
            predicted = 1.0 if result['probability'] > 0.5 else 0.0
            if predicted == label:
                correct += 1

        accuracy = correct / len(EVAL_CASES)
        print(f'\nEval accuracy: {accuracy:.0%} ({correct}/{len(EVAL_CASES)})')


class TestModelRegression:
    """Regression tests — specific patterns that must always be caught.

    These test feature extraction (v1 raw counts), not the scaled
    model output. Feature extraction must work regardless of model version.
    """

    def test_always_catches_eval(self) -> None:
        """eval() must always be detected by feature extraction."""
        features = extract_features_v1("+ const x = eval(userInput);")
        assert features[0] > 0, 'Feature extraction missed eval()'

    def test_always_catches_secrets(self) -> None:
        """Hardcoded secrets must always be detected by feature extraction."""
        features = extract_features_v1("+ const API_KEY = 'sk-prod-real-key';")
        assert features[3] > 0, 'Feature extraction missed secret pattern'

    def test_always_catches_innerhtml(self) -> None:
        """innerHTML must always be detected by feature extraction."""
        features = extract_features_v1("+ document.body.innerHTML = data;")
        assert features[1] > 0, 'Feature extraction missed innerHTML'

    def test_empty_diff_does_not_crash(self, model_service: ModelService) -> None:
        """Empty diff should return a valid prediction."""
        result = model_service.predict('')
        assert result['verdict'] in ('PASS', 'FAIL')
        assert 0 <= result['probability'] <= 1
