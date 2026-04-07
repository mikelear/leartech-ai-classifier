# leartech-ai-classifier

PASS/FAIL code review classifier — a fast pre-filter for the AI review pipeline.

Trained on real feedback data from the leartech code review pipeline. Predicts whether a code diff will pass or fail review in ~10ms on CPU.

## Quick Start

```bash
make setup    # Install UV + dependencies
make all      # Format, lint, typecheck, test
make run      # Start on :8080
```

## API

```
GET  /health       Health check + model status
POST /predict      Predict PASS/FAIL for a code diff
GET  /model/info   Model metadata and training metrics
GET  /docs         OpenAPI documentation (auto-generated)
```

### Predict

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"diff": "+ const API_KEY = \"sk-secret\";\n+ eval(input);"}'
```

```json
{
  "verdict": "FAIL",
  "confidence": 0.92,
  "probability": 0.92,
  "features": {
    "eval_calls": 1,
    "secret_patterns": 1,
    ...
  }
}
```

## Development

| Command | What it does |
|---------|-------------|
| `make setup` | Install UV and all dependencies |
| `make fmt` | Format code with Ruff |
| `make lint` | Lint (Ruff) + type check (mypy strict) |
| `make test` | Run pytest with coverage (80% minimum) |
| `make all` | fmt + lint + test |
| `make check` | all + Docker build |
| `make run` | Run locally with hot reload |
| `make build` | Build Docker image |

## Tooling

| Tool | Purpose |
|------|---------|
| [UV](https://docs.astral.sh/uv/) | Dependency management (fast, modern) |
| [Ruff](https://docs.astral.sh/ruff/) | Linting + formatting (replaces flake8/black/isort) |
| [mypy](https://mypy.readthedocs.io/) | Static type checking (strict mode) |
| [pytest](https://docs.pytest.org/) | Testing with coverage |
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP framework (async, OpenAPI docs) |
| [PyTorch](https://pytorch.org/) | ML inference |

## Pipeline Checks (PR)

| Context | Check | Required |
|---------|-------|----------|
| `pr` | Build + test + preview | Yes |
| `lint` | Ruff format + lint + mypy | Yes |
| `ai-review` | AI code review (Claude + DeepSeek) | No |
| `security-scan` | Gitleaks + Semgrep | No |
| `image-scan` | Grype dependency scan | No |
| `dynamic-scan` | Nuclei + Nikto + Nmap on preview | No |

## Model

- **Architecture:** 3-layer neural network (~1,100 parameters)
- **Input:** 16 hand-crafted features from code diffs
- **Output:** PASS/FAIL probability
- **Training data:** 102 real feedback records from the AI review pipeline
- **File:** `models/code_classifier.pt` (~5KB)

## Project Structure

```
app/
  main.py         FastAPI application
  model.py        Model loading and prediction
  features.py     Feature extraction from diffs
  config.py       Environment-based configuration
tests/
  test_api.py     API integration tests
  test_model.py   Model unit tests
  test_features.py Feature extraction tests
models/
  code_classifier.pt  Trained model
charts/
  leartech-ai-classifier/  Helm chart for K8s deployment
```
# Test
