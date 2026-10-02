# Documentation map

Every number in these documents is generated from `results/raw/` by `scripts/render_docs.py`.

## Start here

| Document | What it is |
|---|---|
| [blog.md](blog.md) | The whole story for a general technical reader, start to finish. |
| [06-results-analysis.md](06-results-analysis.md) | The final results: every bar of the waterfall explained, and the investigation of the pair that collides. |
| [07-performance-model.md](07-performance-model.md) | The model: its equations, calibration, frozen predictions, errors, and how to use it. |
| [interview-prep.md](interview-prep.md) | Twenty likely questions with draft answers, and the numbers to know. |
| [glossary.md](glossary.md) | Every term, one line each, with where it is worked through. |
| [open-questions.md](open-questions.md) | What was measured but not explained, each with the command that would settle it. |

## By topic

AGENTS.md's target structure lists one topic document per area. The learning docs fill that role here; each
has the same shape (intuition, math, worked example, prediction, result, questions, reading).

| Topic (AGENTS.md's name) | Where it is |
|---|---|
| Inference physics (`00`) | [learning/M0-foundations.md](learning/M0-foundations.md), [learning/M1-nanoserve.md](learning/M1-nanoserve.md) |
| Quantization (`01`) | [learning/M3-quant-reference.md](learning/M3-quant-reference.md), [learning/M4-quant-production.md](learning/M4-quant-production.md) |
| KV cache (`02`) | [learning/M5-kv-cache.md](learning/M5-kv-cache.md) |
| Speculative decoding (`03`) | [learning/M6-speculative-decoding.md](learning/M6-speculative-decoding.md) |
| Kernels (`04`) | [04-kernels.md](04-kernels.md), [learning/M7-triton-kernels.md](learning/M7-triton-kernels.md) |
| Benchmark methodology (`05`) | [05-benchmark-methodology.md](05-benchmark-methodology.md), [learning/M2-baselines.md](learning/M2-baselines.md) |
| Results analysis (`06`) | [06-results-analysis.md](06-results-analysis.md), [learning/M8-full-stack.md](learning/M8-full-stack.md) |
| Performance model (`07`) | [07-performance-model.md](07-performance-model.md) |
| Presentation | [learning/M9-presentation.md](learning/M9-presentation.md) |

## By milestone

| | Learning doc | Gate report |
|---|---|---|
| M0 | [foundations](learning/M0-foundations.md) | [report](gates/M0-report.md) |
| M1 | [nanoserve](learning/M1-nanoserve.md) | [report](gates/M1-report.md) |
| M2 | [baselines](learning/M2-baselines.md) | [report](gates/M2-report.md) |
| M3 | [quantization from scratch](learning/M3-quant-reference.md) | [report](gates/M3-report.md) |
| M4 | [quantization in production](learning/M4-quant-production.md) | [report](gates/M4-report.md) |
| M5 | [KV cache](learning/M5-kv-cache.md) | [report](gates/M5-report.md) |
| M6 | [speculative decoding](learning/M6-speculative-decoding.md) | [report](gates/M6-report.md) |
| M7 | [Triton kernels](learning/M7-triton-kernels.md) | [report](gates/M7-report.md) |
| M8 | [the full stack](learning/M8-full-stack.md) | [report](gates/M8-report.md) |
| M9 | [presentation](learning/M9-presentation.md) | [report](gates/M9-report.md) |

Decisions that changed the plan: [decisions/](decisions/). The dated log of what happened, dead ends
included: [`JOURNAL.md`](../JOURNAL.md).
