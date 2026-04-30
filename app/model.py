"""Model loading and prediction."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.features import (
    FEATURE_NAMES,
    extract_all_features,
    load_artefacts,
)


class CodeClassifier(nn.Module):
    """Neural network for PASS/FAIL code review prediction.

    Architecture adapts to the input dimension stored in the checkpoint.
    """

    def __init__(self, input_dim: int = 16) -> None:
        super().__init__()
        # Architecture matches what was trained in Sessions 10.8/11.5
        hidden = 64 if input_dim > 100 else 32
        hidden2 = hidden // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — features in, probability out."""
        result: torch.Tensor = self.net(x)
        return result


class ModelService:
    """Loads and serves the trained model."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model: CodeClassifier | None = None
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load model weights, TF-IDF vectorizer, and scaler."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f'Model file not found: {self.model_path}')

        # Load TF-IDF and scaler artefacts from same directory as model
        models_dir = str(path.parent)
        load_artefacts(models_dir)

        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)  # nosemgrep: pickles-in-pytorch
        input_dim = checkpoint.get('num_features', 16)

        self.model = CodeClassifier(input_dim=input_dim)

        # Handle both key formats: bare Sequential ("0.weight") and
        # wrapped Sequential ("net.0.weight") from different training sessions
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('net.') for k in state_dict):
            self.model.load_state_dict(state_dict)
        else:
            # Bare Sequential keys — add "net." prefix
            prefixed = {f'net.{k}': v for k, v in state_dict.items()}
            self.model.load_state_dict(prefixed)

        self.model.eval()

        self.metadata = {
            'accuracy': checkpoint.get('accuracy', 0),
            'precision': checkpoint.get('precision', 0),
            'recall': checkpoint.get('recall', 0),
            'f1': checkpoint.get('f1', 0),
            'training_examples': checkpoint.get('training_examples', 0),
            'epochs_trained': checkpoint.get('epochs_trained', 0),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'feature_names': checkpoint.get('feature_names', FEATURE_NAMES),
            'num_features': input_dim,
            'eval_accuracy': checkpoint.get('eval_accuracy', 0),
        }

    def predict(self, diff: str, pipeline_signals: dict[str, Any] | None = None) -> dict[str, Any]:
        """Predict PASS/FAIL for a code diff.

        Args:
            diff: The code diff text.
            pipeline_signals: Optional pipeline signal dict (services_affected,
                touches_critical, etc.). Defaults to neutral when not provided.

        Returns:
            Dictionary with verdict, confidence, probability, and feature values.
        """
        if self.model is None:
            return {'verdict': 'ERROR', 'confidence': 0.0, 'probability': 0.0, 'features': {}}

        features = extract_all_features(diff, pipeline_signals)

        with torch.no_grad():
            probability = self.model(features).item()

        verdict = 'FAIL' if probability > 0.5 else 'PASS'
        confidence = probability if probability > 0.5 else 1 - probability

        feature_names = self.metadata.get('feature_names', FEATURE_NAMES)
        feature_values = {}
        for i, name in enumerate(feature_names):
            if i < len(features):
                feature_values[name] = round(features[i].item(), 4)

        return {
            'verdict': verdict,
            'confidence': round(confidence, 4),
            'probability': round(probability, 4),
            'features': feature_values,
        }

    def info(self) -> dict[str, Any]:
        """Return model metadata."""
        return self.metadata
