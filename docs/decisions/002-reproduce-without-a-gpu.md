# ADR 002: Quick reproduce runs without a GPU, and the full reproduction is written but not run

- **Status:** accepted (owner's decision at the M8 gate, 2026-10-02)
- **Date:** 2026-10-02

## Context

AGENTS.md's M9 asks for `scripts/reproduce_all.sh` end to end on a fresh GPU machine, a "quick reproduce"
subset under an hour, and the acceptance test "a fresh clone + Docker + `make reproduce-quick` regenerates the
headline figure".

Three facts about this project change what that should mean:

1. **The month's GPU budget was spent** by the end of M8. A full rerun costs about what the milestones did
   (`results/compute_log.csv`).
2. **There is no local Docker image.** Since ADR 001 all compute runs in Modal's containers, defined in
   `infra/modal_app.py`. The laptop cannot hold an image with PyTorch and vLLM.
3. **The raw records of every run are committed** (`results/raw/`), and every figure, table and dashboard
   number is derived from them by code that needs no GPU and no PyTorch.

## Decision

- **`make reproduce-quick` regenerates everything a reader sees from the committed raw records, on any
  machine, without a GPU or a Modal account.** It needs `uv` and a small dependency group (numpy, matplotlib,
  plotly, pyyaml). It redraws every figure of M0–M8, fails if any redrawn caption differs from the committed
  one, re-renders the tables in the docs, and checks the dashboard's data. Captions are computed from the
  data, so an equal caption means the committed figure shows what the records say today.
- **CI runs that same target on every push.** "A fresh clone regenerates the headline figure" is therefore
  tested continuously, on a machine that has never seen the project, instead of once by hand.
- **`scripts/reproduce_all.sh` is the full reproduction: written, documented, not run for the M9 gate.** It is
  the ordered list of commands that produced `results/raw/`, with what each needs. The owner will run it, or
  parts of it, when there is budget.
- **The open question from M8 is kept as one command** (`make open-question`,
  [docs/open-questions.md](../open-questions.md)) for the same reason.
- **GitHub Pages is prepared, not switched on.** The dashboard is in `site/` and a manually started workflow
  publishes it. Turning Pages on for the repository is the owner's action.

## Consequences

- What is verified is the chain *raw records → everything shown*. What is not verified by M9 is *GPU →
  raw records*: that the measurements come out the same if run again. Two servers were run twice in M8 and
  the stock one repeated closely; that is the evidence there is.
- The quick path proves less than AGENTS.md asked for and proves it more often.
- The topic documents `docs/00`–`03` of AGENTS.md's target structure were not written as separate files: the
  learning docs M1, M3, M5 and M6 cover the same ground with the same structure. `docs/README.md` maps one
  onto the other.
