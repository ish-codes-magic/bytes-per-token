# Shortcuts. Every compute target runs in the cloud through Modal: the laptop only needs `uv`.
# (No `make` on Windows? Run the commands after each target name directly.)
MODAL := uv run --only-group local modal run infra/modal_app.py

.PHONY: setup lint format test test-gpu probe bench-baseline quality-baseline bench-kernels figures docs

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

bench-baseline:  # M2: the load sweep, vLLM's offline engine, and the saturation run (about an hour of L4 time)
	$(MODAL)::m2
	$(MODAL)::m2offline
	$(MODAL)::m2 --config benchmarks/configs/m2_saturation.yaml --out m2_saturation.jsonl

quality-baseline:  # M2: perplexity, lm-eval tasks and the needle grid, one container per (task, model)
	$(MODAL)::m2q

bench-kernels:  # M7: profile, both kernels' microbenchmarks, nanoserve end to end, quality (7 containers)
	$(MODAL)::m7

figures:
	$(MODAL)::figures

docs:
	uv run --only-group local python scripts/render_docs.py
