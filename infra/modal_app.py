"""Modal app: the single door to all compute in this project. Nothing heavy ever runs on the laptop.

From the repo root (the laptop's venv holds only `modal` and `ruff`):

    uv run --only-group local modal run infra/modal_app.py::test        # CPU unit tests
    uv run --only-group local modal run infra/modal_app.py::test_gpu    # GPU-marked tests on an L4
    uv run --only-group local modal run infra/modal_app.py::probe       # M0 hardware probe -> results/raw/
    uv run --only-group local modal run infra/modal_app.py::figures     # figures -> results/figures/
    uv run --only-group local modal run infra/modal_app.py::fetch       # download model weights (once)

The container image is built in the cloud from pyproject.toml + uv.lock and cached after the first build.
Every function has a timeout, so a hung job can never eat the month's free credit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"
GPU = "L4"
CACHE = "/cache"  # a Modal Volume: model weights are downloaded once and reused by every run
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# The laptop has no ML packages, but fastserve's stdlib-only modules (results, report) work here too.
sys.path.insert(0, str(REPO / "src"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(str(REPO), groups=["dev"])
    .env({"PYTHONPATH": f"{REMOTE}/src", "MPLBACKEND": "Agg", "HF_HOME": f"{CACHE}/huggingface"})
    .workdir(REMOTE)
    .add_local_file(REPO / "pyproject.toml", f"{REMOTE}/pyproject.toml")
    .add_local_dir(REPO / "src", f"{REMOTE}/src", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "tests", f"{REMOTE}/tests", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "benchmarks", f"{REMOTE}/benchmarks")  # configs, prompts, model configs
)

app = modal.App("bytes-per-token", image=image)
hf_cache = modal.Volume.from_name("bpt-hf-cache", create_if_missing=True)


def _pytest(args: list[str]) -> int:
    import subprocess

    return subprocess.run([sys.executable, "-m", "pytest", *args], cwd=REMOTE).returncode


@app.function(cpu=2, memory=4096, timeout=15 * 60)
def pytest_cpu(args: list[str]) -> int:
    return _pytest(args)


@app.function(gpu=GPU, cpu=2, memory=16384, timeout=15 * 60, volumes={CACHE: hf_cache})
def pytest_gpu(args: list[str]) -> int:
    return _pytest(args)


@app.function(cpu=2, memory=4096, timeout=20 * 60, volumes={CACHE: hf_cache})
def download_model(repo_id: str) -> str:
    """Download config, tokenizer and safetensors weights into the Volume (skipped if already there)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
    hf_cache.commit()  # make the files visible to other containers
    return path


@app.function(gpu=GPU, cpu=2, memory=8192, timeout=30 * 60)
def hw_probe(config_text: str, config_path: str, git: dict) -> list[dict]:
    import yaml

    from fastserve.hw.probe import run_probe
    from fastserve.results import to_plain

    return to_plain(run_probe(yaml.safe_load(config_text), git=git, config_path=config_path))


@app.function(cpu=2, memory=4096, timeout=10 * 60)
def render_hw_figures(records: list[dict]) -> dict[str, bytes]:
    import tempfile

    from fastserve.viz.hw_figures import make_all

    with tempfile.TemporaryDirectory() as tmp:
        return {path.name: path.read_bytes() for path in make_all(records, tmp)}


@app.local_entrypoint()
def test(args: str = "-q") -> None:
    if code := pytest_cpu.remote(args.split()):
        raise SystemExit(code)


@app.local_entrypoint()
def test_gpu(args: str = "-q -m gpu") -> None:
    if code := pytest_gpu.remote(args.split()):
        raise SystemExit(code)


@app.local_entrypoint()
def fetch(model: str = DEFAULT_MODEL) -> None:
    print(f"{model} is at {download_model.remote(model)} on the {hf_cache.name or 'cache'} volume")


@app.local_entrypoint()
def probe(config: str = "benchmarks/configs/hw_probe.yaml") -> None:
    from fastserve.results import append_jsonl, git_info

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    records = hw_probe.remote((REPO / config).read_text(encoding="utf-8"), config, git)
    out = REPO / "results" / "raw" / "hw_probe.jsonl"
    print(f"wrote {append_jsonl(out, records)} records to {out.relative_to(REPO)}")


@app.local_entrypoint()
def figures() -> None:
    from fastserve.results import latest_run, read_jsonl

    records = latest_run(read_jsonl(REPO / "results" / "raw" / "hw_probe.jsonl"))
    out = REPO / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in render_hw_figures.remote(records).items():
        (out / name).write_bytes(data)
        print(f"wrote results/figures/{name}")
