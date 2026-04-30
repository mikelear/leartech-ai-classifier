"""FastAPI application — PASS/FAIL code review classifier."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import settings
from app.model import ModelService

logger = logging.getLogger(__name__)

model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup, cleanup on shutdown."""
    global model_service  # noqa: PLW0603
    logger.info('Loading model from %s', settings.model_path)
    model_service = ModelService(settings.model_path)
    logger.info(
        'Model loaded: %d parameters, accuracy=%.1f%%',
        model_service.metadata['parameters'],
        model_service.metadata['accuracy'] * 100,
    )
    yield
    logger.info('Shutting down')


app = FastAPI(
    title='leartech-ai-classifier',
    description='PASS/FAIL code review classifier — fast pre-filter for the AI review pipeline',
    version='0.1.0',
    lifespan=lifespan,
)


class PipelineSignals(BaseModel):
    """Optional pipeline signals for enhanced prediction."""

    services_affected: int = 1
    touches_critical: float = 0.0
    unexpected_edges: int = 0
    coverage_gaps: int = 0
    e2e_passed: float = 1.0
    leartech_violations: int = 0


class PredictRequest(BaseModel):
    """Request body for the predict endpoint."""

    diff: str = Field(..., description='The code diff to classify')
    pipeline_signals: PipelineSignals | None = Field(
        None, description='Optional pipeline signals (risk-assessor, e2e, semgrep)',
    )


class PredictResponse(BaseModel):
    """Response from the predict endpoint."""

    verdict: str = Field(..., description='PASS or FAIL')
    confidence: float = Field(..., description='Confidence of the prediction (0-1)')
    probability: float = Field(..., description='Raw probability of FAIL (0-1)')
    features: dict[str, float] = Field(..., description='Extracted feature values')


class HealthResponse(BaseModel):
    """Response from the health endpoint."""

    status: str
    model_loaded: bool
    parameters: int
    accuracy: float


class ModelInfoResponse(BaseModel):
    """Response from the model info endpoint."""

    parameters: int
    training_examples: int
    epochs_trained: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_names: list[str]


@app.get('/health', response_model=HealthResponse)
async def health() -> dict[str, Any]:
    """Health check — reports model status."""
    if model_service is None:
        return {
            'status': 'unhealthy',
            'model_loaded': False,
            'parameters': 0,
            'accuracy': 0.0,
        }
    return {
        'status': 'healthy',
        'model_loaded': True,
        'parameters': model_service.metadata['parameters'],
        'accuracy': model_service.metadata['accuracy'],
    }


@app.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest) -> dict[str, Any]:
    """Predict PASS/FAIL for a code diff."""
    if model_service is None:
        return {
            'verdict': 'ERROR',
            'confidence': 0.0,
            'probability': 0.0,
            'features': {},
        }
    signals = request.pipeline_signals.model_dump() if request.pipeline_signals else None
    return model_service.predict(request.diff, pipeline_signals=signals)


@app.get('/model/info', response_model=ModelInfoResponse)
async def model_info() -> dict[str, Any]:
    """Return model metadata and training metrics."""
    if model_service is None:
        return {
            'parameters': 0,
            'training_examples': 0,
            'epochs_trained': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'feature_names': [],
        }
    return model_service.info()
