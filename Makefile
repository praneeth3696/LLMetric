.PHONY: help setup test generate-data run-backend run-frontend dev smoke-test build clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PYTEST = $(VENV)/bin/pytest
UVICORN = $(VENV)/bin/uvicorn

help:
	@echo "LLMetrics Management Commands:"
	@echo "  make setup          Install Python & Node dependencies, build frontend"
	@echo "  make test           Run the full test suite with pytest"
	@echo "  make generate-data  Run Phase 1 core CLI to generate synthetic data & figures"
	@echo "  make run-backend    Start FastAPI backend server on http://127.0.0.1:8000"
	@echo "  make run-frontend   Start Vite frontend dev server on http://localhost:5173"
	@echo "  make dev            Run backend and frontend concurrently"
	@echo "  make smoke-test     Run end-to-end smoke test against all syllabus endpoints"
	@echo "  make build          Build production bundle for frontend"
	@echo "  make clean          Remove cache directories and temporary files"

setup:
	@echo "==> Setting up Python virtual environment and dependencies..."
	@if [ ! -d "$(VENV)" ]; then uv venv $(VENV) --python 3.11; fi
	uv pip install --python $(PYTHON) -e .
	uv pip install --python $(PYTHON) python-multipart pytest httpx
	@echo "==> Setting up Frontend dependencies..."
	cd frontend && npm install
	@echo "==> Building frontend..."
	cd frontend && npm run build
	@echo "==> Setup complete!"

test:
	$(PYTEST) -v

generate-data:
	$(PYTHON) -m core.run_all --days 90 --volume 1200 --seed 42

run-backend:
	$(UVICORN) backend.main:app --host 127.0.0.1 --port 8000 --reload

run-frontend:
	cd frontend && npm run dev

dev:
	@echo "==> Starting LLMetrics backend and frontend concurrently..."
	@trap 'kill 0' EXIT; \
	$(UVICORN) backend.main:app --host 127.0.0.1 --port 8000 & \
	(cd frontend && npm run dev) & \
	wait

smoke-test:
	$(PYTHON) backend/tests/test_e2e_smoke.py

build:
	cd frontend && npm run build

clean:
	rm -rf .pytest_cache __pycache__ core/**/__pycache__ backend/**/__pycache__
	rm -rf frontend/dist frontend/.vite

