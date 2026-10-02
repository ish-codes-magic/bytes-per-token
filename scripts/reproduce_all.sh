#!/usr/bin/env bash
# Reproduce every measurement of the project from scratch, in the cloud, then redraw everything from it.
#
# NOT run as part of the M9 gate (the owner's decision: the month's GPU budget was spent). It is the list of
# commands that produced results/raw/, in order. To check the figures and tables against the raw results that
# are already committed, without a GPU, use `make reproduce-quick` instead.
#
# Needs:  uv; a Modal account (`uv run --only-group local modal setup`).
# Cost:   every container is billed by Modal. The project's own runs cost what results/compute_log.csv says
#         (the README's "Reproduce" section quotes the total); a clean rerun without the dead ends is less.
# GPU:    one NVIDIA L4 per container. On another GPU the numbers change, and so may the findings: the
#         collision in docs/06-results-analysis.md section 6 is specific to GPUs older than Hopper.
# Models: Qwen3-0.6B and Qwen3-1.7B (not gated). The quantized checkpoints are read from the Hugging Face
#         Hub (ishita-codes-ai/*). To rebuild them instead, set REBUILD_CHECKPOINTS=1.
#
# Usage:  bash scripts/reproduce_all.sh            # everything
#         bash scripts/reproduce_all.sh m5 m8      # only these steps
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
MODAL="uv run --only-group local modal run infra/modal_app.py"
PY="uv run --only-group local python"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "The working tree has uncommitted changes: results would be flagged as dirty. Commit first." >&2
  exit 1
fi

steps=("$@")
wanted() { [[ ${#steps[@]} -eq 0 ]] || [[ " ${steps[*]} " == *" $1 "* ]]; }
step() { echo; echo "==== $1: $2 ===="; }

if wanted m0; then
  step m0 "the GPU's measured ceilings"
  $MODAL::probe
fi
if wanted m1; then
  step m1 "nanoserve: prefill and decode, per-op timing"
  $MODAL::m1
fi
if wanted m2; then
  step m2 "stock vLLM: the load sweep, the offline engine, saturation, and the quality baseline"
  $MODAL::m2
  $MODAL::m2offline
  $MODAL::m2 --config benchmarks/configs/m2_saturation.yaml --out m2_saturation.jsonl
  $MODAL::m2q
fi
if wanted m3; then
  step m3 "quantization from scratch, validated against llm-compressor"
  $MODAL::m3
fi
if wanted m4; then
  step m4 "quantized checkpoints in vLLM"
  if [[ "${REBUILD_CHECKPOINTS:-0}" == "1" ]]; then
    $MODAL::m4   # builds the checkpoints first (64 GB containers), then serves and scores them
  else
    $MODAL::m4 --steps serving,suite,kl,fidelity,gsm8k
  fi
fi
if wanted m5; then
  step m5 "the KV cache: quantization policies, FP8 KV in vLLM, prefix caching"
  $MODAL::m5
fi
if wanted m6; then
  step m6 "speculative decoding: the lossless test's data, drafters, EAGLE-3 in vLLM"
  $MODAL::m6
fi
if wanted m7; then
  step m7 "the two Triton kernels: profile, microbenchmarks, nanoserve, quality"
  $MODAL::test_gpu --args "-q tests/kernels"
  $MODAL::m7
fi
if wanted m8; then
  step m8 "the full stack: 40 servers, the stack's quality, FlashInfer's call costs, the profiles"
  $MODAL::m8
  $MODAL::m8 --quality perplexity,needle,tasks
  $MODAL::m8 --plan
  $MODAL::m8 --profile --only 1.7b-s,1.7b-sg,1.7b-fs,0.6b-w,0.6b-g,0.6b-wg
fi
if wanted render; then
  step render "figures, tables, the dashboard's data, and the compute log"
  $MODAL::figures
  $PY scripts/compute_log.py
  $PY scripts/render_docs.py
  $PY scripts/build_site.py
fi

echo
echo "Done. New records were appended to results/raw/; the reports read the newest run of each."
echo "Predictions in benchmarks/predictions/ were written before the original runs: compare, do not edit."
