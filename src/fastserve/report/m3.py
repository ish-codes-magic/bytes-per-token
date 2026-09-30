"""M3 analysis: quantization records → the numbers and tables the docs use. Stdlib only.

Tasks can be re-run, so for every (model, configuration) the newest record wins.
"""

from __future__ import annotations

from typing import Any

from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]
SMALL, LARGE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"


def newest(records: Records, experiment: str) -> Records:
    """The newest record of each kind: per (model, config) for m3_config, per model otherwise."""
    chosen: dict[tuple, dict] = {}
    for r in sorted((r for r in records if r["experiment"] == experiment), key=lambda r: r["timestamp"]):
        m = r["metrics"]
        chosen[(m.get("model"), m.get("config"), m.get("method"))] = r
    return list(chosen.values())


def configs(records: Records, model: str = SMALL) -> dict[str, dict[str, Any]]:
    """{config name: metrics} for one model."""
    return {
        r["metrics"]["config"]: r["metrics"]
        for r in newest(records, "m3_config")
        if r["metrics"]["model"] == model
    }


def single(records: Records, experiment: str, model: str = SMALL) -> dict[str, Any] | None:
    found = [r["metrics"] for r in newest(records, experiment) if r["metrics"].get("model", model) == model]
    return found[-1] if found else None


def _kl(c: dict[str, dict], name: str) -> float | None:
    return c[name]["mean_kl"] if name in c else None


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def sensitivity_shares(scan: dict[str, Any], n_layers: int) -> dict[str, float]:
    """Share (%) of the summed single-module KL from down_proj, and from the first two and last layers."""
    cells = scan["cells"]
    total = sum(c["kl"] for c in cells)
    edges = {0, 1, n_layers - 1}
    return {
        "down_proj": 100 * sum(c["kl"] for c in cells if c["module"] == "down_proj") / total,
        "edges": 100 * sum(c["kl"] for c in cells if c["layer"] in edges) / total,
    }


def m3_observables(records: Records) -> dict[str, float | None]:
    """The quantities predicted before M3 (benchmarks/predictions/m3.json)."""
    c, big = configs(records), configs(records, LARGE)
    k = lambda name: _kl(c, name)  # noqa: E731
    atlas = single(records, "m3_outliers")
    scan = single(records, "m3_sensitivity")
    shares = sensitivity_shares(scan, 28) if scan else {}
    return {
        "bf16_ppl": c["bf16"]["perplexity_ref"] if "bf16" in c else None,
        "kl_int8_channel": k("rtn-int8-channel"),
        "kl_int4_g128": k("rtn-int4-g128"),
        "channel_over_g128": _ratio(k("rtn-int4-channel"), k("rtn-int4-g128")),
        "kl_int3_g128": k("rtn-int3-g128"),
        "kl_int2_g64": k("rtn-int2-g64-asym"),
        "full_over_textbook": _ratio(k("rtn-int4-g128-full"), k("rtn-int4-g128")),
        "nf4_over_int4": _ratio(k("nf4-b64"), k("rtn-int4-g64")),
        "kl_fp8_weight": k("fp8-weight-channel"),
        "gptq_over_rtn": _ratio(k("gptq-int4-g128"), k("rtn-int4-g128-full")),
        "awq_over_rtn": _ratio(k("awq-int4-g128"), k("rtn-int4-g128-full")),
        "gptq_over_library": _ratio(k("gptq-int4-g128"), k("library-gptq-int4-g128")),
        "awq_over_library": _ratio(k("awq-int4-g128"), k("library-awq-int4-g128")),
        "rotation_gain_channel": _ratio(k("rot-rtn-int4-channel"), k("rtn-int4-channel")),
        "kl_w8a8_fp8_token": k("w8a8-fp8-token"),
        "kl_w8a8_int8_static": k("w8a8-int8-tensor-static"),
        "smoothquant_gain": _ratio(k("sq-w8a8-int8-tensor-static"), k("w8a8-int8-tensor-static")),
        "outlier_ratio": atlas["residual_ratio"] if atlas else None,
        "down_proj_share": shares.get("down_proj"),
        "edge_layer_share": shares.get("edges"),
        "calib_8_over_128": _ratio(k("calib-c4-8"), k("gptq-int4-g128")),
        "calib_code_over_c4": _ratio(k("calib-code"), k("gptq-int4-g128")),
        "calib_wiki_over_c4": _ratio(k("calib-wikitext"), k("gptq-int4-g128")),
        "large_over_small_int4": _ratio(_kl(big, "rtn-int4-g128"), k("rtn-int4-g128")),
    }


# ---- labels and tables --------------------------------------------------------------------------------


def _grid(e: dict[str, Any]) -> str:
    size = {"tensor": "per-tensor", "channel": "per-channel"}.get(e.get("granularity", "group"))
    size = size or f"g{e.get('group_size', 128)}"
    kind = "" if e.get("symmetric", True) else ", asymmetric"
    kind += ", full range" if e.get("full_range") else ""
    return f"INT{e['bits']} {size}{kind}"


def label(e: dict[str, Any]) -> str:
    """A readable name for one configuration entry."""
    prefix = "rotated, " if e.get("rotate") else ("γ folded, " if e.get("fold") else "")
    suffix = ", FP32 copy" if e.get("dtype") == "float32" else ""
    if e["method"] == "bf16":
        return ("BF16 (reference)" if not prefix else f"BF16, {prefix[:-2]}") + suffix
    return prefix + _method_label(e) + suffix


def _method_label(e: dict[str, Any]) -> str:
    method = e["method"]
    if method == "rtn":
        return f"RTN {_grid(e)}"
    if method == "nf4":
        return f"NF4, blocks of {e['block_size']}"
    if method == "fp8_weight":
        return f"FP8 E4M3 weights, per-{e.get('granularity', 'channel')}"
    if method == "gptq":
        extra = ", true-sequential" if e.get("true_sequential") else ""
        data, samples = e.get("calibration"), e.get("samples")
        if data or samples:
            extra += f", calibrated on {data or 'c4'}" + (f" × {samples}" if samples else "")
        return f"GPTQ {_grid(e)}{extra}"
    if method == "awq":
        scaling = "duo scaling" if e.get("duo_scaling", True) else "paper scaling"
        return f"AWQ {_grid(e)} ({scaling}{', clip search' if e.get('clip') else ''})"
    if method == "library":
        return f"llm-compressor {e['checkpoint'].rsplit('-', 1)[-1].upper()} {_grid(e)}"
    if method == "w8a8":
        acts = f"{'static' if e.get('static') else 'dynamic'} per-{e['act']} activations"
        smooth = "SmoothQuant + " if e.get("smooth") is not None else ""
        return f"{smooth}W8A8 {e['format'].upper()}, {acts}"
    return e["name"]


def _fmt_kl(x: float) -> str:
    return f"{x:.2e}" if 0 < x < 0.01 else f"{x:.3f}"


def config_table(records: Records, names: list[str], model: str = SMALL, reference: str | None = None) -> str:
    """One row per configuration: storage, then how far it moved the model."""
    c = configs(records, model)
    rows = []
    for name in names:
        if name not in c:
            continue
        m = c[name]
        ppl = m["perplexity_cand"] / m["perplexity_ref"] - 1
        row = [
            label(m["entry"]),
            f"{m['bits_per_weight']:.2f}",
            f"{m['model_gb']:.2f}",
            _fmt_kl(m["mean_kl"]),
            f"{m['top1_agreement']:.1%}",
            f"{m['perplexity_cand']:.2f} ({ppl:+.1%})",
        ]
        if reference:
            ref = c.get(reference, {}).get("mean_kl")
            row.append(f"{m['mean_kl'] / ref:.2f}×" if ref else "—")
        rows.append(row)
    headers = [
        "Configuration",
        "Bits per weight",
        "Model (GB)",
        "KL vs BF16",
        "Top-1 agreement",
        "Perplexity",
    ]
    if reference:
        headers.append(f"KL vs {label(c[reference]['entry'])}" if reference in c else "Relative KL")
    return markdown_table(headers, rows)


def names_in_task(records: Records, task: str, model: str = SMALL) -> list[str]:
    """Configuration names of one task, in the order the config lists them."""
    runs = [
        r
        for r in newest(records, "m3_config")
        if r["config"].get("task") == task and r["metrics"]["model"] == model
    ]
    return [r["metrics"]["config"] for r in sorted(runs, key=lambda r: r["timestamp"])]


def sensitivity_table(records: Records, n_layers: int = 28) -> str:
    """KL from quantizing one module at a time, summed per module type; the five worst modules."""
    scan = single(records, "m3_sensitivity")
    if not scan:
        return ""
    cells, total = scan["cells"], sum(c["kl"] for c in scan["cells"])
    kinds = {}
    for cell in cells:
        kinds[cell["module"]] = kinds.get(cell["module"], 0.0) + cell["kl"]
    rows = [
        [kind, _fmt_kl(kl), f"{kl / total:.0%}"] for kind, kl in sorted(kinds.items(), key=lambda kv: -kv[1])
    ]
    top = sorted(cells, key=lambda c: -c["kl"])[:5]
    worst = ", ".join(f"layer {c['layer']} {c['module']} ({_fmt_kl(c['kl'])})" for c in top)
    table = markdown_table(
        ["Module type", f"Summed KL ({scan['spec']}, one module at a time)", "Share"], rows
    )
    return f"{table}\n\nMost sensitive single modules: {worst}."


def worked_example_block(records: Records) -> str:
    """The 4-weight GPTQ example, step by step, as markdown."""
    ex = single(records, "m3_worked_example", model=None)
    if not ex:
        return ""

    def vec(v: list[float]) -> str:
        return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"

    lines = [
        f"Weights w = {vec(ex['w'])}, grid {ex['spec']} (step {max(abs(x) for x in ex['w']) / 3:.2f}).",
        "",
        "H = 2 XᵀX / n from six input samples; inputs 0 and 1 move together, 2 and 3 in opposite directions:",
        "",
        "```",
        *(vec(row) for row in ex["H"]),
        "```",
        "",
        "| Step | Column rounded | Its rounding error | Weights after the update |",
        "|---|---|---|---|",
    ]
    for i, step in enumerate(ex["steps"], start=1):
        lines.append(f"| {i} | {step['column']} | {step['error'][0]:+.3f} | {vec(step['weights'])} |")
    lines += [
        "",
        f"- RTN: {vec(ex['rtn'])}, output error ½ΔHΔᵀ = {ex['loss_rtn']:.4f}",
        f"- GPTQ: {vec(ex['gptq'])}, output error {ex['loss_gptq']:.4f} "
        f"({ex['loss_rtn'] / ex['loss_gptq']:.1f}× smaller)",
    ]
    return "\n".join(lines)


def model_size_table(records: Records, names: list[str]) -> str:
    """The same configurations on Qwen3-0.6B and Qwen3-1.7B: how the damage changes with model size."""
    small, large = configs(records, SMALL), configs(records, LARGE)
    rows = []
    for name in names:
        if name not in small or name not in large:
            continue
        a, b = small[name], large[name]
        ratio = f"{b['mean_kl'] / a['mean_kl']:.2f}×" if a["mean_kl"] else "—"
        rows.append(
            [
                label(a["entry"]),
                f"{a['perplexity_ref']:.2f} → {a['perplexity_cand']:.2f}",
                _fmt_kl(a["mean_kl"]),
                f"{b['perplexity_ref']:.2f} → {b['perplexity_cand']:.2f}",
                _fmt_kl(b["mean_kl"]),
                ratio,
            ]
        )
    headers = [
        "Configuration",
        "0.6B perplexity",
        "0.6B KL",
        "1.7B perplexity",
        "1.7B KL",
        "KL, 1.7B / 0.6B",
    ]
    return markdown_table(headers, rows)
