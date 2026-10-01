"""M6, the reference side: speculative decoding in nanoserve, on real prompts and real models.

Tasks (benchmarks/configs/m6_spec.yaml, `tasks`, one container each):
- agreement: the target's greedy output for every prompt, then how each drafter would have fared at every
             draft length k: one teacher-forced drafter pass gives per-position agreement, and the replay in
             spec/simulate.py turns it into tokens per target pass, acceptance rates and histograms. The
             target can be BF16 or one of M4's quantized checkpoints (the interaction experiment).
- loop:      the actual draft-verify loop on a few prompts: its output must equal plain greedy decoding, and
             its rounds must equal the replay's.
- lossless:  sampling at temperature 1: speculative samples against the target's exact first-token
             distribution (chi-square), and against plain samples at later positions (total variation, with
             a plain-vs-plain noise floor).
- timing:    one decode step of the drafter and the target, and the target verifying k + 1 tokens at once:
             the two costs every speedup formula needs.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import torch

from fastserve.engine.generate import generate
from fastserve.engine.kv_cache import ContiguousKVCache
from fastserve.engine.loader import load_pretrained, model_dir
from fastserve.engine.model import CausalLM, use_fused_attention
from fastserve.engine.sampler import SamplingParams
from fastserve.quality.prompts import task_prompts
from fastserve.spec.drafters import ModelDrafter, NGramDrafter, NoDrafter
from fastserve.spec.generate import speculative_generate
from fastserve.spec.lm import CachedLM
from fastserve.spec.simulate import from_draft, model_rounds, ngram_rounds, run_rates, summarize
from fastserve.spec.stats import chi_square, chi_square_limit, total_variation
from fastserve.timing import cuda_time_ms
from fastserve.timing import summarize as timing_summary

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def load_model(spec: dict[str, Any], model_name: str) -> CausalLM:
    """A BF16 model from the Hub cache, or one of M4's compressed checkpoints (`compressed`, `act`)."""
    if "compressed" not in spec:
        model = load_pretrained(model_dir(spec.get("model", model_name)))
    else:
        from fastserve.engine.config import ModelConfig
        from fastserve.engine.loader import from_state_dict
        from fastserve.quant.compressed import load_dense
        from fastserve.quant.model import w8a8_model
        from fastserve.quant.w8a8 import W8A8Config

        source = spec["compressed"]
        folder = source if source.startswith("/") else model_dir(source, download=True)
        cfg = ModelConfig.from_pretrained_json(f"{model_dir(spec.get('model', model_name))}/config.json")
        model = from_state_dict(cfg, load_dense(folder, device="cuda"), device="cuda", dtype=torch.bfloat16)
        if spec.get("act"):  # W8A8: activations quantized on the fly, as vLLM does
            w8a8_model(model, W8A8Config(spec["format"], act_granularity=spec["act"]), quantize_weights=False)
    return use_fused_attention(model)


@torch.inference_mode()
def greedy_outputs(
    model: CausalLM, prompts: list[list[int]], params: SamplingParams, batch: int
) -> list[list[int]]:
    """The model's own greedy continuation of every prompt (static batches, longest prompts together)."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: dict[int, list[int]] = {}
    for start in range(0, len(order), batch):
        ids = order[start : start + batch]
        for i, out in zip(ids, generate(model, [prompts[i] for i in ids], params), strict=True):
            outputs[i] = out
    return [outputs[i] for i in range(len(prompts))]


@torch.inference_mode()
def agreement_bits(draft: CausalLM, prompt: list[int], output: list[int]) -> list[bool]:
    """Per output position j: does the drafter, given the true context, greedily predict output[j]?"""
    ids = torch.tensor(prompt + output[:-1], device=draft.lm_head.weight.device)[None]
    predicted = draft(ids)[0, len(prompt) - 1 :].argmax(dim=-1)  # [len(output)]
    return (predicted.cpu() == torch.tensor(output)).tolist()


def agreement(
    target: CausalLM,
    drafts: dict[str, CausalLM | NGramDrafter],
    prompts: dict[str, list[list[int]]],
    params: SamplingParams,
    ks: list[int],
    batch: int,
) -> list[dict[str, Any]]:
    """One row per (drafter, task) plus an "all" row per drafter: replayed speculation at every k."""
    rows = []
    flat = [(task, p) for task, ps in prompts.items() for p in ps]
    outputs = greedy_outputs(target, [p for _, p in flat], params, batch)
    for name, drafter in drafts.items():
        per_task: dict[str, list[tuple[list[int], list[int], list[bool] | None]]] = {}
        for (task, prompt), output in zip(flat, outputs, strict=True):
            bits = None if isinstance(drafter, NGramDrafter) else agreement_bits(drafter, prompt, output)
            per_task.setdefault(task, []).append((prompt, output, bits))
        everything = [item for group in per_task.values() for item in group]
        for task, items in [*per_task.items(), ("all", everything)]:
            tokens = sum(len(o) for _, o, _ in items)
            by_k = {}
            for k in ks:
                rounds = [
                    ngram_rounds(p, o, k, drafter) if bits is None else model_rounds(bits, k)
                    for p, o, bits in items
                ]
                by_k[str(k)] = summarize(rounds, tokens)
            row = {"drafter": name, "task": task, "prompts": len(items), "tokens": tokens, "by_k": by_k}
            if items[0][2] is not None:
                row["rates"] = run_rates([b for _, _, bits in items for b in bits])
            rows.append(row)
    return rows


def highlight(
    tokenizer: Any,
    drafter: CausalLM | NGramDrafter,
    target: CausalLM,
    prompt: list[int],
    params: SamplingParams,
    k: int,
) -> dict[str, Any]:
    """One output as text pieces with a flag each: drafted and accepted, or written by the target."""
    output = greedy_outputs(target, [prompt], params, 1)[0]
    if isinstance(drafter, NGramDrafter):
        rounds = ngram_rounds(prompt, output, k, drafter)
    else:
        rounds = model_rounds(agreement_bits(drafter, prompt, output), k)
    pieces = [tokenizer.decode([t]) for t in output]
    return {"k": k, "pieces": pieces, "from_draft": from_draft(rounds, len(output))}


def loop_check(
    target: CausalLM, draft: CausalLM, prompts: list[list[int]], params: SamplingParams, k: int
) -> dict[str, Any]:
    """Run the real loop: is its output the plain greedy output, and do its rounds match the replay?"""
    identical = rounds_equal = 0
    actual_tokens = actual_rounds = replay_rounds = 0
    first_divergence = []
    for prompt in prompts:
        max_len = len(prompt) + params.max_new_tokens + k + 2
        plain = speculative_generate(CachedLM(target, max_len), NoDrafter(), prompt, params, 0)
        spec = speculative_generate(
            CachedLM(target, max_len), ModelDrafter(CachedLM(draft, max_len), params), prompt, params, k
        )
        replay = model_rounds(agreement_bits(draft, prompt, plain.tokens), k)
        identical += spec.tokens == plain.tokens
        rounds_equal += [a for _, a in spec.rounds[:-1]] == [a for _, a in replay[: len(spec.rounds) - 1]]
        if spec.tokens != plain.tokens:
            first_divergence.append(
                next(i for i, (a, b) in enumerate(zip(spec.tokens, plain.tokens, strict=False)) if a != b)
            )
        actual_tokens += len(spec.tokens)
        actual_rounds += len(spec.rounds)
        replay_rounds += len(replay)
    return {
        "prompts": len(prompts),
        "k": k,
        "identical_outputs": identical,
        "rounds_match_replay": rounds_equal,
        "first_divergence": first_divergence,
        "tokens_per_round_actual": actual_tokens / actual_rounds,
        "tokens_per_round_replay": actual_tokens / replay_rounds,
    }


@torch.inference_mode()
def first_token_paths(
    target: CausalLM, prompt: list[int], k: int, plain_batch: int
) -> dict[str, torch.Tensor]:
    """The first generated token's distribution, as three differently shaped passes compute it. [vocab] each.

    Mathematically they are one distribution. Numerically each pass rounds differently, and in BF16 (whose
    values near 20 are 0.125 apart) that can move real probability when two tokens are nearly tied.
    - "reference": the prompt alone, one sequence, no cache.
    - "plain": what plain sampling uses: a batch of identical prompts prefilled through a cache.
    - "speculative": what the verifier uses: the last prompt token plus k drafted tokens, on a cached prompt.
    """
    weight = target.lm_head.weight
    device, n = weight.device, len(prompt)
    ids = torch.tensor([prompt], device=device)

    def distribution(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=-1).cpu()

    cache = ContiguousKVCache(
        target.config, max_batch=plain_batch, max_len=n + 1, dtype=weight.dtype, device=device
    )
    positions = torch.arange(n, device=device).expand(plain_batch, -1)
    last = torch.full((plain_batch,), n - 1, device=device)
    batched = target(ids.expand(plain_batch, -1), positions, cache, select=last)[0]
    lm = CachedLM(target, n + k + 2)
    lm.logits_after(prompt + [0] * k, k + 1)  # the first sample's pass caches the prompt
    verifier = lm.logits_after(prompt + [1] * k, k + 1)[0]  # every later sample's pass looks like this one
    return {
        "reference": distribution(target(ids)[0, -1]),
        "plain": distribution(batched),
        "speculative": distribution(verifier),
    }


def lossless(
    target: CausalLM,
    draft: CausalLM,
    prompt: list[int],
    samples: int,
    tokens: int,
    k: int,
    seed: int = 0,
    plain_batch: int = 50,
) -> dict[str, Any]:
    """Sampling at temperature 1: do speculative samples follow the target's distribution?

    - First token, chi-square against an exact distribution: each sampler against the distribution its own
      pass computed ("own"), and against the single-sequence reference pass. A correct sampler passes the
      first. It passes the second only if the passes agree numerically, which `path_tv` measures directly.
    - Longer prefixes: total variation between speculative and plain samples of the first t tokens, next to
      the distance between two independent plain samples (the noise floor at this sample size).
    """
    params = SamplingParams(max_new_tokens=tokens, temperature=1.0)
    max_len = len(prompt) + tokens + k + 2
    generator = torch.Generator().manual_seed(seed)
    target_lm, drafter = CachedLM(target, max_len), ModelDrafter(CachedLM(draft, max_len), params)
    spec = [
        speculative_generate(target_lm, drafter, prompt, params, k, generator).tokens for _ in range(samples)
    ]

    device = target.lm_head.weight.device
    on_device = torch.Generator(device=device).manual_seed(seed)  # plain sampling draws where the model is
    plain = []
    for _ in range(2):  # two independent plain samples: their distance is the noise floor
        batches = -(-samples // plain_batch)  # small batches: each copy of the prompt holds its own KV cache
        runs = [generate(target, [prompt] * plain_batch, params, generator=on_device) for _ in range(batches)]
        plain.append([o for run in runs for o in run][:samples])

    paths = first_token_paths(target, prompt, k, plain_batch)
    exact = {name: {i: p for i, p in enumerate(dist.tolist()) if p > 0} for name, dist in paths.items()}
    first = {"speculative": Counter(o[0] for o in spec), "plain": Counter(o[0] for o in plain[0])}
    statistic, dof = chi_square(first["speculative"], exact["reference"])
    positions = []
    for t in range(tokens):
        counts = [Counter(tuple(o[: t + 1]) for o in group) for group in (spec, plain[0], plain[1])]
        positions.append(
            {
                "tokens": t + 1,  # the joint distribution of the first t + 1 tokens
                "tv_spec_vs_plain": total_variation(counts[0], counts[1]),
                "tv_plain_vs_plain": total_variation(counts[1], counts[2]),
            }
        )
    return {
        "samples": samples,
        "k": k,
        "chi_square": statistic,  # against the reference pass
        "chi_square_plain": chi_square(first["plain"], exact["reference"])[0],
        "chi_square_own": chi_square(first["speculative"], exact["speculative"])[0],  # against its own pass
        "chi_square_plain_own": chi_square(first["plain"], exact["plain"])[0],
        "dof": dof,
        "limit": chi_square_limit(dof),
        "path_tv": {  # how far apart the passes' first-token distributions are: no sampling involved
            "plain_vs_reference": 0.5 * (paths["plain"] - paths["reference"]).abs().sum().item(),
            "speculative_vs_reference": 0.5 * (paths["speculative"] - paths["reference"]).abs().sum().item(),
            "speculative_vs_plain": 0.5 * (paths["speculative"] - paths["plain"]).abs().sum().item(),
        },
        "top_probability": paths["reference"].max().item(),
        "prefixes": positions,
    }


@torch.inference_mode()
def step_times(
    model: CausalLM, context: list[int], verify_sizes: list[int], iters: int = 30
) -> dict[str, Any]:
    """Median time (ms) of a forward pass over q new tokens after `context`, for each q: q = 1 is a decode
    step, q = k + 1 is verifying k drafted tokens."""
    weight = model.lm_head.weight
    cache = ContiguousKVCache(
        model.config,
        max_batch=1,
        max_len=len(context) + max(verify_sizes) + 1,
        dtype=weight.dtype,
        device="cuda",
    )
    ids = torch.tensor(context, device="cuda")[None]
    model(ids, torch.arange(len(context), device="cuda")[None], cache)  # prefill the context once
    times = {}
    for q in verify_sizes:
        new = torch.randint(0, 1000, (1, q), device="cuda")
        positions = torch.arange(len(context), len(context) + q, device="cuda")[None]
        times[str(q)] = timing_summary(
            cuda_time_ms(
                lambda new=new, positions=positions: model(new, positions, cache), warmup=5, iters=iters
            )
        ).median_ms
    return {"context": len(context), "ms": times}


def run_task(
    task: str, config: dict[str, Any], *, run_id: str, git: dict | None, config_path: str
) -> list[dict]:
    """Run one section of benchmarks/configs/m6_spec.yaml's `tasks` in this container."""
    from transformers import AutoTokenizer

    from fastserve.results import environment_info, make_record

    spec, env = config["tasks"][task], environment_info()
    meta = {"path": config_path, "task": task, "reference": config["reference"]}
    ref = config["reference"]

    def record(experiment: str, metrics: dict[str, Any]) -> dict:
        return make_record(experiment, metrics, run_id=run_id, config=meta, git=git, env=env)

    tokenizer = AutoTokenizer.from_pretrained(model_dir(ref["target"]))
    stop = tuple(tokenizer.convert_tokens_to_ids(list(STOP_TOKENS)))
    greedy = SamplingParams(max_new_tokens=ref["max_new_tokens"], stop_token_ids=stop)
    prompts = {t: task_prompts(tokenizer, t, ref["prompts_per_task"]) for t in ref["tasks"]}
    target = load_model(spec.get("target", {}), ref["target"])
    target_name = spec.get("target", {}).get("name", "bf16")
    start = time.perf_counter()

    if spec["kind"] == "agreement":
        drafts: dict[str, CausalLM | NGramDrafter] = {}
        for d in spec["drafts"]:
            entry = config["drafters"][d]
            drafts[d] = (
                NGramDrafter(entry["max_n"], entry["min_n"])
                if entry.get("ngram")
                else load_model(entry, ref["draft"])
            )
        rows = agreement(target, drafts, prompts, greedy, ref["ks"], ref["batch"])
        records = [record("m6_agreement", {"target": target_name, **row}) for row in rows]
        for d in spec.get("highlight", []):
            for t in ref["tasks"]:
                shown = highlight(tokenizer, drafts[d], target, prompts[t][0], greedy, ref["highlight_k"])
                records.append(
                    record("m6_highlight", {"target": target_name, "drafter": d, "task": t, **shown})
                )
    else:
        draft = load_model(config["drafters"][spec["draft"]], ref["draft"])
        if spec["kind"] == "loop":
            some = [p for t in ref["tasks"] for p in prompts[t][: spec["prompts_per_task"]]]
            params = SamplingParams(max_new_tokens=spec["max_new_tokens"], stop_token_ids=stop)
            records = []
            for dtype in spec.get("dtypes", ["bfloat16"]):
                # float32 is the control: if outputs differ in BF16 only, the cause is rounding at near-ties
                target.to(getattr(torch, dtype))
                draft.to(getattr(torch, dtype))
                check = loop_check(target, draft, some, params, spec["k"])
                names = {"target": target_name, "drafter": spec["draft"], "dtype": dtype}
                records.append(record("m6_loop_check", {**names, **check}))
        elif spec["kind"] == "lossless":
            records = []
            for dtype in spec.get("dtypes", ["bfloat16"]):  # float32: the control for BF16's rounding
                target.to(getattr(torch, dtype))
                draft.to(getattr(torch, dtype))
                batch = spec["plain_batch"][dtype]
                for t in spec.get("tasks", ref["tasks"]):
                    metrics = lossless(
                        target,
                        draft,
                        prompts[t][0],
                        spec["samples"],
                        spec["tokens"],
                        spec["k"],
                        plain_batch=batch,
                    )
                    names = {"target": target_name, "drafter": spec["draft"], "task": t, "dtype": dtype}
                    records.append(record("m6_lossless", {**names, **metrics}))
                    print(f"lossless {dtype} {t}: {metrics['path_tv']}", flush=True)
        elif spec["kind"] == "timing":
            context = max(prompts["summarize"], key=len)
            sizes = [1, *(k + 1 for k in ref["ks"])]
            records = [
                record(
                    "m6_step_times",
                    {"model": ref["target"], "role": "target", **step_times(target, context, sizes)},
                ),
                record(
                    "m6_step_times",
                    {"model": ref["draft"], "role": "draft", **step_times(draft, context, [1])},
                ),
            ]
        else:
            raise ValueError(f"unknown task kind {spec['kind']!r}")
    print(f"{task}: {len(records)} records ({time.perf_counter() - start:.0f} s)", flush=True)
    return records
