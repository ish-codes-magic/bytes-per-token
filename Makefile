# Shortcuts. Every compute target runs in the cloud through Modal: the laptop only needs `uv`.
# (No `make` on Windows? Run the commands after each target name directly.)
MODAL := uv run --only-group local modal run infra/modal_app.py

.PHONY: setup lint format test test-gpu probe figures docs

setup:
	uv sync --only-group local
	uv run --only-group local modal setup

lint:
	uv run --only-group local ruff check .
	uv run --only-group local ruff format --check .

format:
	uv run --only-group local ruff format .
	uv run --only-group local ruff check --fix .

test:
	$(MODAL)::test

test-gpu:
	$(MODAL)::test_gpu

probe:
	$(MODAL)::probe

figures:
	$(MODAL)::figures

docs:
	uv run --only-group local python scripts/render_docs.py
