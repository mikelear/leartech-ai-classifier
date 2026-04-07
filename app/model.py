"""Model loading and prediction."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.features import NUM_FEATURES, FEATURE_NAMES, extract_features


class CodeClassifier(nn.Module):
    """Neural network for PASS/FAIL code review prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — features in, probability out."""
        return self.net(x)


class ModelService:
    """Loads and serves the trained model."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = CodeClassifier()
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load model weights and metadata from .pt file."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f'Model file not found: {self.model_path}')

        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        }

    def predict(self, diff: str) -> dict[str, Any]:
        """Predict PASS/FAIL for a code diff.

        Args:
            diff: The code diff text.

        Returns:
            Dictionary with verdict, confidence, probability, and feature values.
        """
        features = extract_features(diff)

        with torch.no_grad():
            probability = self.model(features).item()

        verdict = 'FAIL' if probability > 0.5 else 'PASS'
        confidence = probability if probability > 0.5 else 1 - probability

        feature_values = {
            name: features[i].item()
            for i, name in enumerate(FEATURE_NAMES)
        }

        return {
            'verdict': verdict,
            'confidence': round(confidence, 4),
            'probability': round(probability, 4),
            'features': feature_values,
        }

    def info(self) -> dict[str, Any]:
        """Return model metadata."""
        return self.metadata
