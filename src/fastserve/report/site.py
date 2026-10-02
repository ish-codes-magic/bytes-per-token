"""The dashboard's data: everything `site/` shows, assembled from results/raw/ and the frozen predictions.

`scripts/build_site.py` writes the result to `site/data/dashboard.json`. The page draws its charts from it
and runs the serving model in the browser (`site/model.js`, a port of perfmodel/serving.py). The `checks`
list is how the port is kept honest: inputs with the Python model's outputs, which the page and the tests
recompute in JavaScript and compare. Stdlib only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastserve.engine.config import ModelConfig
from fastserve.perfmodel.serving import (
    FLASH_ATTN,
    FLASHINFER,
    Calibration,
    Hardware,
    Load,
    Speculation,
    Stack,
    linear_params,
    predict,
)
from fastserve.report import m8
from fastserve.results import read_jsonl
from fastserve.serving.ablation import BASE, letters

Records = list[dict[str, Any]]
MODELS = (m8.LARGE, m8.SMALL)
TABLES = {  # blocks of scripts/render_docs.py the page shows as tables, with their headings
    "m8_best_large": "Best measured stack per workload, Qwen3-1.7B",
    "m8_best_small": "Best measured stack per workload, Qwen3-0.6B",
    "m8_quality": "Quality of the lossy parts of the stack, in vLLM",
    "m8_energy": "Energy, Qwen3-1.7B",
    "m8_model_errors": "The frozen predictions against M8",
    "m8_model_informed": "The model with M8's measured inputs, and with the host term",
    "m8_calibration": "The model's calibrated constants",
    "m8_host_chain": "Every server on piecewise CUDA graphs, one user",
}


def rounded(value: Any, digits: int = 7) -> Any:
    """Floats to `digits` significant digits, recursively: a JSON file that is the same on every platform."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    return value


def model_sizes(cfg: ModelConfig) -> dict[str, float]:
    """What the serving model needs to know about a model, as plain numbers for the browser."""
    return {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "linear_params": linear_params(cfg),
        "kv_bytes_per_token_bf16": cfg.kv_bytes_per_token(2),
    }


def stack_dict(stack: Stack) -> dict[str, Any]:
    return asdict(stack)


def servers(records: Records, model: str) -> dict[str, Any]:
    """Every server of one model: how it started, and what it did on each workload."""
    out: dict[str, Any] = {}
    for label in m8.labels_run(records, model):
        start = m8.graphs_of(records, model, label)
        loads = {}
        for workload in m8.WORKLOADS:
            m = m8.serving(records, model, label, workload)
            if not m:
                continue
            loads[workload] = {
                "tok_s": m["summary"]["output_throughput"],
                "tpot_ms": m8.stat(records, model, label, workload, "tpot_ms"),
                "ttft_ms": m8.stat(records, model, label, workload, "ttft_ms"),
                "power_w": m8.power(records, model, label, workload),
                "step_ms": m8.step_ms(records, model, label, workload),
                "kept": m8.tokens_per_pass(m) if "s" in letters(label) else 1.0,
            }
        out[label] = {
            "graph": m8.graph_mode(start),
            "deployable": label in m8.candidates(records, model),
            "workloads": loads,
        }
    return out


def interactions(records: Records, model: str) -> list[dict[str, Any]]:
    rows = []
    for a, b in m8.pairs():
        values = {w: m8.interaction(records, model, a, b, w) for w in m8.WORKLOADS[:4]}
        rows.append(
            {"pair": f"{m8.TECHNIQUES[a]} + {m8.TECHNIQUES[b]}", "label": m8.label_pair(a, b), **values}
        )
    return rows


def host_chain(records: Records) -> list[dict[str, Any]]:
    """Each piecewise server's step time at one user, next to its closest full-graph server's."""
    rows = []
    for model in (m8.SMALL, m8.LARGE):
        for label in m8.labels_run(records, model):
            if label.endswith("-r2") or m8.graph_mode(m8.graphs_of(records, model, label)) != "piecewise":
                continue
            twin = next(
                (t for t in m8.FULL_GRAPH_TWINS.get(label, []) if m8.step_ms(records, model, t)), None
            )
            if twin:
                rows.append(
                    {
                        "model": model,
                        "label": label,
                        "twin": twin,
                        "piecewise_ms": m8.step_ms(records, model, label),
                        "full_ms": m8.step_ms(records, model, twin),
                        "power_w": m8.power(records, model, label, "m8_latency"),
                    }
                )
    return rows


def _grid(title: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    """A needle grid as a matrix: rows are depths, columns are lengths, values the share of secrets found."""
    lengths = sorted({c["length"] for c in cells})
    depths = sorted({c["depth"] for c in cells})
    passed = [
        [
            sum(c["passed"] for c in cells if (c["length"], c["depth"]) == (length, depth))
            / max(1, sum(1 for c in cells if (c["length"], c["depth"]) == (length, depth)))
            for length in lengths
        ]
        for depth in depths
    ]
    return {"title": title, "lengths": lengths, "depths": depths, "passed": passed}


def needle_grids(m2_quality: Records, m5: Records, m7: Records, records: Records) -> list[dict[str, Any]]:
    """Every needle-in-a-haystack grid measured in the project, newest of each kind."""
    found: dict[str, dict[str, Any]] = {}

    def keep(title: str, metrics: dict[str, Any]) -> None:
        found[title] = _grid(title, metrics["cells"])  # a later record of the same kind replaces an earlier

    for source, experiment, name in (
        (m2_quality, "quality_needle", lambda m: "BF16 weights and cache, vLLM (M2)"),
        (m5, "m5_vllm_needle", lambda m: "FP8 KV cache, vLLM (M5)"),
        (m5, "m5_kv_needle", lambda m: f"KV policy `{m['policy']}`, nanoserve (M5)"),
        (m7, "m7_needle", lambda m: "4-bit KV codes read by kernel 2, nanoserve (M7)"),
        (records, "m8_needle", lambda m: "FP8 weights + FP8 KV cache, vLLM (M8)"),
    ):
        for record in sorted(
            (r for r in source if r["experiment"] == experiment), key=lambda r: r["timestamp"]
        ):
            metrics = record["metrics"]
            keep(f"{metrics['model'].split('/')[-1]}: {name(metrics)}", metrics)
    return list(found.values())


CHECK_LOADS = [  # (users, prompt, output): the corners and the middle of what the calculator offers
    (1, 512, 256),
    (8, 2048, 256),
    (64, 512, 128),
    (96, 4096, 256),
    (500, 16384, 256),
    (1, 32768, 64),
]
CHECK_STACKS = [  # (weights, FP8 KV, speculation, prefix caching)
    ("bf16", False, False, False),
    ("fp8", False, False, False),
    ("int4", False, True, False),
    ("fp8", True, False, False),
    ("fp8", True, True, False),
    ("fp8", False, True, True),
    ("fp8", True, True, True),
    ("bf16", True, False, True),
]


def make_stack(
    weights: str, fp8_kv: bool, spec: bool, prefix: bool, kept: float, draft_bytes: float
) -> Stack:
    return Stack(
        weights=weights,
        kv="fp8" if fp8_kv else "bf16",
        backend=FLASHINFER if fp8_kv else FLASH_ATTN,
        prefix_caching=prefix,
        speculation=Speculation(3, kept, draft_bytes) if spec else None,
    )


def checks(
    configs: dict[str, ModelConfig],
    hw: Hardware,
    calibrations: dict[str, Calibration],
    starts: dict[str, Any],
    draft_bytes: dict[str, float],
    kept: float,
) -> list[dict[str, Any]]:
    """Inputs with the Python model's outputs, for the JavaScript port to reproduce."""
    cases = []
    for model, cfg in configs.items():
        for name, cal in calibrations.items():
            for weights, fp8_kv, spec, prefix in CHECK_STACKS:
                stack = make_stack(weights, fp8_kv, spec, prefix, kept, draft_bytes[model])
                capacity = m8.kv_capacity(model, stack, cfg, starts)
                for users, prompt, output in CHECK_LOADS:
                    load = Load(users=users, prompt_len=prompt, output_len=output, kv_tokens=capacity)
                    if prefix:  # half the prompt cached, shared by four sequences
                        load = replace(load, cached_len=prompt / 2, shared_len=prompt / 2, sharers=4.0)
                    got = predict(cfg, hw, stack, cal, load)
                    expected = {
                        key: got[key] for key in ("batch", "tok_s", "tpot_ms", "step_ms", "prefill_ms")
                    }
                    cases.append(
                        {
                            "model": model,
                            "calibration": name,
                            "stack": stack_dict(stack),
                            "load": asdict(load),
                            "kv_capacity": capacity,
                            "expected": expected,
                        }
                    )
    return cases


def map_picks(
    cfg: ModelConfig,
    hw: Hardware,
    cal: Calibration,
    starts: dict[str, Any],
    model: str,
    kept: float,
    draft_bytes: float,
) -> dict[str, Any]:
    """The recommendation map as Python computes it, for the page's own map to be checked against."""
    picks: dict[str, list[list[str]]] = {"fp8": [], "any": []}
    for users in m8.MAP_USERS:
        rows: dict[str, list[str]] = {"fp8": [], "any": []}
        for context in m8.MAP_CONTEXTS:
            got = m8.predict_candidates(cfg, hw, cal, starts, model, users, context, kept, draft_bytes)
            fp8 = [label for label in got if not label.startswith("a")]
            rows["fp8"].append(max(fp8, key=lambda label: got[label]["tok_s"]))
            rows["any"].append(max(got, key=lambda label: got[label]["tok_s"]))
        for key in picks:
            picks[key].append(rows[key])
    return {"users": m8.MAP_USERS, "contexts": m8.MAP_CONTEXTS, "output_len": 256, "picks": picks}


def dashboard(repo: str | Path, blocks: dict[str, str]) -> dict[str, Any]:
    """Everything the page needs. `blocks` are the generated tables of scripts/render_docs.py."""
    import yaml

    repo = Path(repo)
    raw = repo / "results" / "raw"
    records = read_jsonl(raw / "m8_ablation.jsonl")
    m4, m5 = read_jsonl(raw / "m4_production.jsonl"), read_jsonl(raw / "m5_kv.jsonl")
    m7, m2_quality = read_jsonl(raw / "m7_kernels.jsonl"), read_jsonl(raw / "m2_quality.jsonl")
    config = yaml.safe_load(
        (repo / "benchmarks" / "configs" / "m8_ablation.yaml").read_text(encoding="utf-8")
    )
    frozen = json.loads((repo / "benchmarks" / "predictions" / "m8_model.json").read_text(encoding="utf-8"))
    configs = {
        model: ModelConfig.from_pretrained_json(
            repo / "benchmarks" / "models" / f"{model.split('/')[-1]}.config.json"
        )
        for model in MODELS
    }
    cal = m8.calibration_of(frozen)
    calibrations = {"frozen": cal, "informed": m8.informed_calibration(cal, records)}
    head = config["techniques"]["s"]["head"]
    draft_bytes = {model: m8.eagle_bytes(cfg, head) for model, cfg in configs.items()}
    kept = frozen["expected_loads"]["spec_mixed"]["tokens_per_pass"]
    start = next(r for r in records if r["experiment"] == "serving")
    data = {
        "price_per_hour": config["dollars_per_hour"],
        "environment": {
            key: start["env"].get(key) for key in ("gpu", "vllm", "torch", "flashinfer_python", "triton")
        },
        "findings": m8.findings(records, frozen, m4, m2_quality).splitlines(),
        "techniques": m8.TECHNIQUES,
        "ladder": m8.LADDER,
        "workloads": {
            workload: {"label": m8.WORKLOAD_LABELS[workload], **frozen["expected_loads"][workload]}
            for workload in m8.WORKLOADS
        },
        "models": {model: model_sizes(cfg) for model, cfg in configs.items()},
        "hardware": frozen["hardware"],
        "calibrations": {name: asdict(c) for name, c in calibrations.items()},
        "startup_memory": frozen["startup_memory"],
        "draft_bytes": draft_bytes,
        "tokens_per_pass": kept,
        "servers": {model: servers(records, model) for model in MODELS},
        "best": {
            model: {
                workload: {
                    "fp8": m8.best(records, model, workload, allow_int4=False),
                    "any": m8.best(records, model, workload),
                }
                for workload in m8.WORKLOADS
                if m8.tok_s(records, model, BASE, workload)
            }
            for model in MODELS
        },
        "interactions": interactions(records, m8.LARGE),
        "interaction_workloads": m8.WORKLOADS[:4],
        "host_chain": host_chain(records),
        "predictions": [
            {key: row[key] for key in ("model", "label", "workload", "tok_s", "measured")}
            for row in m8.prediction_errors(records, frozen)
        ],
        "needles": needle_grids(m2_quality, m5, m7, records),
        "tables": [
            {"title": title, "markdown": blocks[name]} for name, title in TABLES.items() if name in blocks
        ],
    }
    data = rounded(data)
    # The checks are computed from the rounded constants the page will load, so the browser's model and
    # Python's see exactly the same inputs.
    rounded_cal = Calibration(**data["calibrations"]["informed"])
    data["map"] = map_picks(
        configs[m8.LARGE],
        Hardware(**data["hardware"]),
        rounded_cal,
        data["startup_memory"],
        m8.LARGE,
        data["tokens_per_pass"],
        data["draft_bytes"][m8.LARGE],
    )
    data["checks"] = rounded(
        checks(
            configs,
            Hardware(**data["hardware"]),
            {name: Calibration(**values) for name, values in data["calibrations"].items()},
            data["startup_memory"],
            data["draft_bytes"],
            data["tokens_per_pass"],
        ),
        digits=12,
    )
    return data
