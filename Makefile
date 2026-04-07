# leartech-ai-classifier — Python Gold Standard
#
# Prerequisites: UV installed (curl -LsSf https://astral.sh/uv/install.sh | sh)

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "  leartech-ai-classifier"
	@echo "  ======================"
	@echo ""
	@echo "  Development:"
	@echo "    setup       Install UV and dependencies"
	@echo "    fmt         Format code (ruff)"
	@echo "    lint        Lint code (ruff + mypy)"
	@echo "    test        Run tests with coverage"
	@echo "    all         fmt + lint + test"
	@echo "    check       all + build (pre-push validation)"
	@echo ""
	@echo "  Run:"
	@echo "    run         Run locally on :8080"
	@echo "    build       Build Docker image"
	@echo "    docker-run  Build and run in Docker"
	@echo ""

setup:
	@command -v uv >/dev/null 2>&1 || { echo "Installing UV..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv sync --dev
	@echo "Setup complete. Run 'make all' to validate."

fmt:
	uv run ruff format app tests
	uv run ruff check app tests --select I --fix

lint:
	uv run ruff format --check app tests
	uv run ruff check app tests
	uv run mypy app

test:
	uv run coverage run -m pytest -v
	uv run coverage report

all: fmt lint test

check: all build

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

build:
	docker build -t leartech-ai-classifier .

docker-run: build
	docker run --rm -p 8080:8080 leartech-ai-classifier

.PHONY: help setup fmt lint test all check run build docker-run
