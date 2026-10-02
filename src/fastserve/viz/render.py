"""Draw every milestone's figures from results/raw/. Needs numpy, matplotlib, plotly and pyyaml: no GPU.

This is the whole of "quick reproduce": the raw records are committed, and everything a reader sees is
derived from them. `scripts/make_figures.py` runs it on any machine; the Modal app runs the same function in
a container.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MILESTONES = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
# One campaign per file: only its newest run is drawn.
LATEST_RUN = {
    "hw": "hw_probe.jsonl",
    "m1": "m1_nanoserve.jsonl",
    "m2": "m2_serving.jsonl",
    "m2sat": "m2_saturation.jsonl",
}
# Tasks that can be re-run one at a time: every run is read, and the reports keep the newest result.
EVERY_RUN = {
    "m3": "m3_quant.jsonl",
    "m4": "m4_production.jsonl",
    "m5": "m5_kv.jsonl",
    "m6": "m6_spec.jsonl",
    "m7": "m7_kernels.jsonl",
    "m8": "m8_ablation.jsonl",
}
NEEDS = {"m0": "hw"}  # the raw key a milestone cannot be drawn without, where it is not the milestone's own


def load_raw(raw_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Every raw file that exists, keyed as the figure code expects."""
    from fastserve.report.m2 import newest_per_model
    from fastserve.results import latest_run, read_jsonl

    raw_dir = Path(raw_dir)
    raw = {
        key: latest_run(read_jsonl(raw_dir / name))
        for key, name in LATEST_RUN.items()
        if (raw_dir / name).exists()
    }
    quality = raw_dir / "m2_quality.jsonl"
    if quality.exists():  # gathered from one container per (task, model): keep the newest of each
        raw["m2q"] = newest_per_model(read_jsonl(quality))
    raw.update(
        {key: read_jsonl(raw_dir / name) for key, name in EVERY_RUN.items() if (raw_dir / name).exists()}
    )
    return raw


def available(raw: dict[str, list[dict[str, Any]]]) -> list[str]:
    """The milestones whose records are present."""
    return [ms for ms in MILESTONES if NEEDS.get(ms, ms) in raw]


def _model_config(repo: Path, model: str) -> Any:
    from fastserve.engine.config import ModelConfig

    return ModelConfig.from_pretrained_json(
        repo / "benchmarks" / "models" / f"{model.split('/')[-1]}.config.json"
    )


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render(
    milestone: str, raw: dict[str, list[dict[str, Any]]], repo: str | Path, out_dir: str | Path
) -> list[Path]:
    """Draw one milestone's figures into `out_dir`. `repo` is where the configs and model sizes live."""
    repo = Path(repo)
    configs_dir = repo / "benchmarks" / "configs"
    if milestone == "m0":
        from fastserve.viz import hw_figures

        return hw_figures.make_all(raw["hw"], out_dir)
    if milestone == "m1":
        from fastserve.viz import m1_figures

        cfg = _model_config(repo, raw["m1"][0]["config"]["model"])
        return m1_figures.make_all(raw["m1"], raw["hw"], cfg, out_dir)
    if milestone == "m2":
        from fastserve.hw.analysis import measured_bandwidth
        from fastserve.viz import m2_figures

        written = m2_figures.make_all(raw["m2"], out_dir)
        if raw.get("m2q"):
            written += m2_figures.make_quality(raw["m2q"], out_dir)
        if raw.get("m2sat"):
            configs = {model: _model_config(repo, model) for model in MODELS}
            written += m2_figures.make_saturation(
                raw["m2sat"], configs, measured_bandwidth(raw["hw"]), out_dir
            )
        return written
    if milestone == "m3":
        from fastserve.viz import m3_figures

        return m3_figures.make_all(raw["m3"], out_dir)
    if milestone == "m4":
        from fastserve.hw.analysis import measured_bandwidth, measured_peak_flops
        from fastserve.viz import m4_figures

        configs = {model: _model_config(repo, model) for model in MODELS}
        peaks = measured_peak_flops(raw["hw"])
        hw = {"bandwidth": measured_bandwidth(raw["hw"]), "bf16": peaks["bf16"], "fp8": peaks["fp8"]}
        return m4_figures.make_all(raw["m4"], configs, hw, out_dir)
    if milestone == "m5":
        from fastserve.viz import m5_figures

        config = _yaml(configs_dir / "m5_kv.yaml")
        workloads = _yaml(repo / config["workloads_file"])
        cfg = _model_config(repo, MODELS[0])
        return m5_figures.make_all(raw["m5"], config["policies"], cfg, workloads, out_dir)
    if milestone == "m6":
        from fastserve.viz import m6_figures

        return m6_figures.make_all(raw["m6"], out_dir)
    if milestone == "m7":
        from fastserve.hw.analysis import measured_bandwidth
        from fastserve.viz import m7_figures

        price = _yaml(configs_dir / "m5_kv.yaml")["dollars_per_hour"]  # an L4 hour, as in M5's cost tables
        cfg = _model_config(repo, MODELS[0])
        return m7_figures.make_all(raw["m7"], measured_bandwidth(raw["hw"]), cfg, price, out_dir)
    if milestone == "m8":
        from fastserve.report import m8 as m8_report
        from fastserve.viz import m8_figures

        config = _yaml(configs_dir / "m8_ablation.yaml")
        frozen = json.loads(
            (repo / "benchmarks" / "predictions" / "m8_model.json").read_text(encoding="utf-8")
        )
        configs = {model: _model_config(repo, model) for model in MODELS}
        perplexity = {
            model: m8_report.perplexities(raw["m8"], raw["m5"], raw["m4"], model) for model in configs
        }
        projected = {}
        for model in configs:
            found = m8_report.kernel_projection(raw["m8"], raw["m7"], configs, model)
            if found:
                projected[model] = found["tok_s"]
        return m8_figures.make_all(
            raw["m8"],
            frozen,
            configs,
            m8_report.hardware(raw["hw"]),
            config["dollars_per_hour"],
            perplexity,
            config["techniques"]["s"]["head"],
            projected,
            out_dir,
        )
    raise ValueError(f"unknown milestone {milestone!r}")


def render_all(repo: str | Path, out_dir: str | Path, milestones: list[str] | None = None) -> list[Path]:
    """Every figure of the given milestones (default: all that have records) from `repo`/results/raw."""
    raw = load_raw(Path(repo) / "results" / "raw")
    written: list[Path] = []
    for milestone in milestones or available(raw):
        written += render(milestone, raw, repo, out_dir)
    return written


def stale_captions(written: list[Path], committed_dir: str | Path) -> list[str]:
    """Names of freshly drawn figures whose caption differs from the committed one (or has none committed).

    A caption is computed from the data, so equal captions mean the committed figure shows what the raw
    records say today.
    """
    stale = []
    for path in written:
        if path.name.endswith(".caption.txt"):
            committed = Path(committed_dir) / path.name
            fresh = path.read_text(encoding="utf-8").strip()
            if not committed.exists() or committed.read_text(encoding="utf-8").strip() != fresh:
                stale.append(path.name.removesuffix(".caption.txt"))
    return stale
