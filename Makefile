# Shortcuts. Every compute target runs in the cloud through Modal: the laptop only needs `uv`.
# (No `make` on Windows? Run the commands after each target name directly.)
MODAL := uv run --only-group local modal run infra/modal_app.py

.PHONY: setup lint format test test-gpu probe bench-baseline quality-baseline bench-kernels bench-ablation figures docs site reproduce-quick reproduce-all open-question

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

bench-ablation:  # M8: every server of the ablation plan and its controls (40 containers), then the stack's quality
	$(MODAL)::m8
	$(MODAL)::m8 --quality perplexity,needle,tasks
	$(MODAL)::m8 --plan

figures:
	$(MODAL)::figures

docs:
	uv run --only-group local python scripts/render_docs.py

site:  # the dashboard's data, from results/raw/ (the page itself is static: serve site/ over HTTP)
	uv run --only-group local python scripts/build_site.py

# Redraw every figure from the committed raw results and check each caption against the committed one, then
# check that the tables in the docs and the dashboard's data are what those results give. No GPU, no Modal
# account, about a minute. This is what CI's `reproduce-quick` job runs.
reproduce-quick:
	uv run --only-group quick python scripts/make_figures.py --out reproduced --check
	uv run --only-group quick python scripts/render_docs.py
	uv run --only-group quick python scripts/build_site.py --check
	git diff --stat --exit-code -- README.md docs

reproduce-all:  # every measurement again, in the cloud. Billed GPU time: read scripts/reproduce_all.sh first
	bash scripts/reproduce_all.sh

# The question M8 left open (docs/open-questions.md): why do FP8 weights double the host's time per step on
# piecewise CUDA graphs? Two short profiles on the 1.7B model (a few minutes of L4 each), then their difference.
open-question:
	$(MODAL)::m8 --profile --only 1.7b-wg,1.7b-g
	uv run --only-group local python scripts/compare_profiles.py 1.7b-wg 1.7b-g
