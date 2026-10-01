"""M6 analysis: speculative decoding in nanoserve (agreement, losslessness) and in vLLM (speed). Stdlib only.

Records come from results/raw/m6_spec.jsonl. vLLM servers are labeled `<target>-<method>[-k<n>]`, e.g.
`bf16-draft-k3`, `awq-none`.
"""

from __future__ import annotations

from typing import Any

from fastserve.report.tables import markdown_table
from fastserve.spec.simulate import expected_tokens

Records = list[dict[str, Any]]
DASH = "—"
TASKS = ("chat", "code", "math", "summarize")
TASK_LABELS = {
    "chat": "Chat",
    "code": "Code",
    "math": "Math",
    "summarize": "Summarization",
    "all": "All tasks",
}
DRAFTER_LABELS = {
    "small": "Qwen3-0.6B",
    "small-awq": "Qwen3-0.6B, INT4 (AWQ)",
    "ngram": "N-gram lookup",
}
METHOD_LABELS = {
    "none": "No speculation",
    "draft": "Qwen3-0.6B drafter",
    "draftawq": "INT4 Qwen3-0.6B drafter",
    "ngram": "N-gram lookup",
    "eagle3": "EAGLE-3 head",
}
TARGET_LABELS = {"bf16": "BF16", "fp8": "FP8 W8A8", "awq": "INT4 W4A16 (AWQ)"}
USERS = (4, 16, 64)  # the mixed-task loads


def _newest(records: Records, experiment: str) -> list[dict[str, Any]]:
    return [
        r["metrics"] for r in sorted(records, key=lambda r: r["timestamp"]) if r["experiment"] == experiment
    ]


def _one(records: Records, experiment: str, **match: Any) -> dict[str, Any] | None:
    found = [m for m in _newest(records, experiment) if all(m.get(k) == v for k, v in match.items())]
    return found[-1] if found else None


def _f(x: float | None, digits: int = 2, suffix: str = "") -> str:
    return DASH if x is None else f"{x:,.{digits}f}{suffix}"


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def _mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def parse_label(label: str) -> tuple[str, str, int | None]:
    """ "bf16-draft-k3" → ("bf16", "draft", 3); "awq-none" → ("awq", "none", None)."""
    target, method, *rest = label.split("-")
    return target, method, int(rest[0][1:]) if rest else None


def server_label(method: str, k: int | None) -> str:
    """The display name of a method and draft length, e.g. "Qwen3-0.6B drafter, k = 3"."""
    return METHOD_LABELS.get(method, method) + (f", k = {k}" if k else "")


# ---- nanoserve --------------------------------------------------------------------------------------------


def agreement(m6: Records, target: str, drafter: str, task: str = "all") -> dict[str, Any] | None:
    return _one(m6, "m6_agreement", target=target, drafter=drafter, task=task)


def tokens_per_pass(m6: Records, target: str, drafter: str, task: str, k: int) -> float | None:
    row = agreement(m6, target, drafter, task)
    return row["by_k"][str(k)]["tokens_per_round"] if row and str(k) in row["by_k"] else None


def agree_rate(m6: Records, target: str, drafter: str, task: str = "all") -> float | None:
    row = agreement(m6, target, drafter, task)
    return row["rates"]["agree"] if row and "rates" in row else None


# ---- vLLM -------------------------------------------------------------------------------------------------


def serving(m6: Records, label: str, workload: str, users: int = 1) -> dict[str, Any] | None:
    found = [
        m
        for m in _newest(m6, "serving")
        if m["server"] == label and m["workload"] == workload and m["load"].get("concurrency") == users
    ]
    return found[-1] if found else None


def tok_s(m6: Records, label: str, workload: str, users: int = 1) -> float | None:
    m = serving(m6, label, workload, users)
    return m["summary"].get("output_throughput") if m else None


def speedup(m6: Records, label: str, workload: str, users: int = 1) -> float | None:
    """Tokens/s with speculation ÷ the same target without it, on the same workload and load."""
    target = parse_label(label)[0]
    return _ratio(tok_s(m6, label, workload, users), tok_s(m6, f"{target}-none", workload, users))


def one_user_speedup(m6: Records, label: str) -> float | None:
    """The mean of the four per-task speedups at one user."""
    return _mean([speedup(m6, label, f"spec_{task}") for task in TASKS])


def counters(m6: Records, label: str, workload: str, users: int = 1) -> dict[str, float] | None:
    """vLLM's speculative-decoding counters over one load: drafts, proposed and accepted tokens."""
    m = serving(m6, label, workload, users)
    if not m or "server_timeline" not in m:
        return None
    timeline = m["server_timeline"]
    cols = {c: i for i, c in enumerate(timeline["columns"])}
    if "spec_drafts" not in cols:
        return None
    out = {}
    for name in ("spec_drafts", "spec_draft_tokens", "spec_accepted"):
        values = [r[cols[name]] for r in timeline["rows"] if r[cols[name]] is not None]
        if not values:
            return None
        out[name] = values[-1] - values[0]
    out["per_position"] = timeline.get("spec_accepted_per_position")
    return out


def vllm_acceptance(m6: Records, label: str, workloads: tuple[str, ...]) -> dict[str, float] | None:
    """Acceptance rate and tokens per target pass from vLLM's counters, summed over `workloads`."""
    totals = {"spec_drafts": 0.0, "spec_draft_tokens": 0.0, "spec_accepted": 0.0}
    for workload in workloads:
        c = counters(m6, label, workload)
        if not c:
            return None
        for name in totals:
            totals[name] += c[name]
    if not totals["spec_drafts"] or not totals["spec_draft_tokens"]:
        return None
    return {
        "acceptance_rate": totals["spec_accepted"] / totals["spec_draft_tokens"],
        "tokens_per_round": 1 + totals["spec_accepted"] / totals["spec_drafts"],
    }


def identical_outputs(m6: Records, label: str, workloads: tuple[str, ...]) -> float | None:
    """Share of requests whose generated text has the same checksum as without speculation."""
    target, same, total = parse_label(label)[0], 0, 0
    for workload in workloads:
        spec, base = serving(m6, label, workload), serving(m6, f"{target}-none", workload)
        if not spec or not base:
            return None
        crcs = []
        for m in (spec, base):
            cols = {c: i for i, c in enumerate(m["requests"]["columns"])}
            if "output_crc" not in cols:
                return None
            crcs.append({r[cols["id"]]: r[cols["output_crc"]] for r in m["requests"]["rows"]})
        shared = set(crcs[0]) & set(crcs[1])
        same += sum(crcs[0][i] == crcs[1][i] for i in shared)
        total += len(shared)
    return same / total if total else None


ONE_USER = tuple(f"spec_{task}" for task in TASKS)


def m6_observables(m6: Records) -> dict[str, float | None]:
    """The measured value behind each prediction in benchmarks/predictions/m6.json."""
    all_small = agreement(m6, "bf16", "small")
    rates = (all_small or {}).get("rates") or {}
    agree = rates.get("agree")
    e3 = tokens_per_pass(m6, "bf16", "small", "all", 3)
    code, chat = agree_rate(m6, "bf16", "small", "code"), agree_rate(m6, "bf16", "small", "chat")
    target_times = (_one(m6, "m6_step_times", role="target") or {}).get("ms") or {}
    loop = loop_check(m6, "bfloat16")
    lossless = _newest(m6, "m6_lossless")
    newest = {m["task"]: m for m in lossless}.values()  # the latest run per prompt
    replay = (all_small or {}).get("by_k", {}).get("3", {}).get("acceptance_rate")
    vllm = vllm_acceptance(m6, "bf16-draft-k3", ONE_USER)
    return {
        "agree_all": agree,
        "agree_code_minus_chat": code - chat if code is not None and chat is not None else None,
        "burstiness": rates["after_agree"] - rates["after_miss"] if rates else None,
        "e3_all": e3,
        "e3_over_iid": _ratio(e3, expected_tokens(agree, 3) if agree is not None else None),
        "ngram_e6_summarize": tokens_per_pass(m6, "bf16", "ngram", "summarize", 6),
        "ngram_e6_code_over_chat": _ratio(
            tokens_per_pass(m6, "bf16", "ngram", "code", 6), tokens_per_pass(m6, "bf16", "ngram", "chat", 6)
        ),
        "awq_drafter_agreement": _ratio(agree_rate(m6, "bf16", "small-awq"), agree),
        "fp8_target_agreement": _ratio(agree_rate(m6, "fp8", "small"), agree),
        "awq_target_agreement": _ratio(agree_rate(m6, "awq", "small"), agree),
        "verify4_cost": _ratio(target_times.get("4"), target_times.get("1")),
        "loop_identical": loop["identical_outputs"] / loop["prompts"] if loop else None,
        "lossless_worst": max((m["chi_square"] / m["limit"] for m in newest), default=None),
        "tv_ratio_worst": max(
            (
                p["tv_spec_vs_plain"] / p["tv_plain_vs_plain"]
                for m in newest
                for p in m["prefixes"]
                if p["tv_plain_vs_plain"]
            ),
            default=None,
        ),
        "vllm_accept_vs_replay": vllm["acceptance_rate"] - replay if vllm and replay is not None else None,
        "speedup_draft_k1": one_user_speedup(m6, "bf16-draft-k1"),
        "speedup_draft_k3": one_user_speedup(m6, "bf16-draft-k3"),
        "speedup_draft_k5": one_user_speedup(m6, "bf16-draft-k5"),
        "speedup_draftawq_over_draft": _ratio(
            one_user_speedup(m6, "bf16-draftawq-k3"), one_user_speedup(m6, "bf16-draft-k3")
        ),
        "speedup_ngram_summarize": speedup(m6, "bf16-ngram-k6", "spec_summarize"),
        "speedup_ngram_chat": speedup(m6, "bf16-ngram-k6", "spec_chat"),
        "speedup_eagle3": one_user_speedup(m6, "bf16-eagle3-k3"),
        "speedup_draft_k3_b16": speedup(m6, "bf16-draft-k3", "spec_mixed", 16),
        "speedup_draft_k3_b64": speedup(m6, "bf16-draft-k3", "spec_mixed", 64),
        "speedup_ngram_b64": speedup(m6, "bf16-ngram-k3", "spec_mixed", 64),
        "speedup_fp8_target": one_user_speedup(m6, "fp8-draft-k3"),
        "speedup_awq_target": one_user_speedup(m6, "awq-draft-k3"),
        "vllm_identical_outputs": identical_outputs(m6, "bf16-draft-k3", ONE_USER),
    }


# ---- tables -----------------------------------------------------------------------------------------------


def agreement_table(m6: Records, target: str = "bf16", ks: tuple[int, ...] = (1, 3, 5, 8)) -> str:
    """Per drafter and task: how often the drafter is right, and the tokens each target pass then yields."""
    rows = []
    for drafter in DRAFTER_LABELS:
        for task in (*TASKS, "all"):
            row = agreement(m6, target, drafter, task)
            if not row:
                continue
            rates = row.get("rates") or {}
            rows.append(
                [
                    DRAFTER_LABELS[drafter],
                    TASK_LABELS[task],
                    _f(100 * rates["agree"], 1, "%") if rates else DASH,
                    _f(100 * rates["after_agree"], 1, "%") if rates else DASH,
                    _f(100 * rates["after_miss"], 1, "%") if rates else DASH,
                    *(_f(tokens_per_pass(m6, target, drafter, task, k)) for k in ks),
                ]
            )
    headers = ["Drafter", "Task", "Agreement", "After an agreement", "After a miss"]
    return markdown_table([*headers, *(f"Tokens per pass, k = {k}" for k in ks)], rows)


def lossless_table(m6: Records) -> str:
    """Temperature 1 on the real models: chi-square on the first token, and TV on the whole sampled prefix."""
    rows, length = [], 0
    for m in {m["task"]: m for m in _newest(m6, "m6_lossless")}.values():
        length = len(m["prefixes"])
        last = m["prefixes"][-1]
        rows.append(
            [
                TASK_LABELS[m["task"]],
                _f(m["samples"], 0),
                _f(m["chi_square"], 0),
                _f(m["chi_square_plain"], 0),
                _f(m["limit"], 0),
                _f(last["tv_spec_vs_plain"], 3),
                _f(last["tv_plain_vs_plain"], 3),
            ]
        )
    headers = [
        "Prompt",
        "Samples",
        "χ², speculative",
        "χ², plain sampling",
        "5σ limit",
        f"TV over {length} tokens, speculative vs plain",
        "TV, plain vs plain",
    ]
    return markdown_table(headers, rows)


def step_table(m6: Records) -> str:
    """nanoserve: a target pass over q new tokens against one over 1, and the drafter's step."""
    target, draft = _one(m6, "m6_step_times", role="target"), _one(m6, "m6_step_times", role="draft")
    if not target:
        return markdown_table(["New tokens in the pass", "Target pass (ms)", "÷ a 1-token pass"], [])
    base = target["ms"]["1"]
    rows = [
        [q, _f(ms), _f(ms / base, 2, "×")]
        for q, ms in sorted(target["ms"].items(), key=lambda kv: int(kv[0]))
    ]
    if draft:
        rows.append(["drafter, 1 token", _f(draft["ms"]["1"]), _f(draft["ms"]["1"] / base, 2, "×")])
    return markdown_table(["New tokens in the pass", "Pass (ms)", "÷ the target's 1-token pass"], rows)


def one_user_table(m6: Records) -> str:
    """vLLM, one user: speedup per task for every BF16-target method, with vLLM's own acceptance."""
    labels = sorted({m["server"] for m in _newest(m6, "serving") if m["server"].startswith("bf16-")})
    rows = []
    for label in sorted(
        labels, key=lambda x: (parse_label(x)[1] != "none", parse_label(x)[1], parse_label(x)[2] or 0)
    ):
        _, method, k = parse_label(label)
        accept = vllm_acceptance(m6, label, ONE_USER)
        same = identical_outputs(m6, label, ONE_USER)
        rows.append(
            [
                server_label(method, k),
                *(_f(speedup(m6, label, f"spec_{task}"), 2, "×") for task in TASKS),
                _f(one_user_speedup(m6, label), 2, "×"),
                _f(100 * accept["acceptance_rate"], 0, "%") if accept else DASH,
                _f(accept["tokens_per_round"]) if accept else DASH,
                _f(100 * same, 0, "%") if same is not None and method != "none" else DASH,
            ]
        )
    headers = [
        "Method",
        *(TASK_LABELS[t] for t in TASKS),
        "Mean",
        "Draft tokens accepted",
        "Tokens per target pass",
        "Outputs identical to no speculation",
    ]
    return markdown_table(headers, rows)


def batch_table(m6: Records) -> str:
    """vLLM, mixed tasks: tokens/s without speculation, and each method's speedup, as users are added."""
    labels = {m["server"] for m in _newest(m6, "serving") if m["server"].startswith("bf16-")}
    rows = []
    for label in sorted(
        labels, key=lambda x: (parse_label(x)[1] != "none", parse_label(x)[1], parse_label(x)[2] or 0)
    ):
        _, method, k = parse_label(label)
        if method == "none":
            cells = [_f(tok_s(m6, label, "spec_mixed", u), 0) + " tok/s" for u in USERS]
            rows.append([server_label(method, k), _f(1.0, 2, "×"), *cells])
        else:
            cells = [_f(speedup(m6, label, "spec_mixed", u), 2, "×") for u in USERS]
            rows.append([server_label(method, k), _f(one_user_speedup(m6, label), 2, "×"), *cells])
    return markdown_table(["Method", "1 user", *(f"{u} users" for u in USERS)], rows)


def interaction_table(m6: Records) -> str:
    """The same drafter against BF16, FP8 and INT4 targets: agreement (nanoserve) and speed (vLLM)."""
    rows = []
    for target in TARGET_LABELS:
        base = _mean([tok_s(m6, f"{target}-none", w) for w in ONE_USER])
        accept = vllm_acceptance(m6, f"{target}-draft-k3", ONE_USER)
        rows.append(
            [
                TARGET_LABELS[target],
                _f(100 * agree_rate(m6, target, "small"), 1, "%")
                if agree_rate(m6, target, "small")
                else DASH,
                _f(tokens_per_pass(m6, target, "small", "all", 3)),
                _f(base, 0),
                _f(100 * accept["acceptance_rate"], 0, "%") if accept else DASH,
                _f(one_user_speedup(m6, f"{target}-draft-k3"), 2, "×"),
            ]
        )
    headers = [
        "Target",
        "Drafter agreement (nanoserve)",
        "Tokens per pass, k = 3 (replay)",
        "Tokens/s without speculation (vLLM)",
        "Draft tokens accepted (vLLM)",
        "Speedup with the drafter, k = 3",
    ]
    return markdown_table(headers, rows)


def loop_check(m6: Records, dtype: str) -> dict[str, Any] | None:
    """The real-loop check in one dtype (records from before the float32 control count as BF16)."""
    found = [m for m in _newest(m6, "m6_loop_check") if m.get("dtype", "bfloat16") == dtype]
    return found[-1] if found else None


def loop_table(m6: Records) -> str:
    """The real draft-verify loop against plain greedy decoding and the replay, in BF16 and in float32."""
    rows = []
    for dtype, name in (("bfloat16", "BF16"), ("float32", "float32 (control)")):
        m = loop_check(m6, dtype)
        if not m:
            continue
        rows.append(
            [
                name,
                _f(m["prompts"], 0),
                f"{m['identical_outputs']} of {m['prompts']}",
                f"{m['rounds_match_replay']} of {m['prompts']}",
                _f(m["tokens_per_round_actual"]),
                _f(m["tokens_per_round_replay"]),
            ]
        )
    headers = [
        "Arithmetic",
        "Prompts",
        "Outputs identical to plain greedy",
        "Rounds identical to the replay",
        "Tokens per pass, real loop",
        "Tokens per pass, replay",
    ]
    return markdown_table(headers, rows)
