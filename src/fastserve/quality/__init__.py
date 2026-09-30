"""Quality: did an optimization make the model worse? Speed is never reported without these.

perplexity.py  perplexity on fixed text, and per-token KL divergence vs a reference model (teacher-forced)
needle.py      needle-in-a-haystack: can the model retrieve a fact from deep inside a long context?
tasks.py       a small lm-evaluation-harness suite (GSM8K, MMLU, HumanEval) through vLLM
"""
