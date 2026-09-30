"""Modal app: the single door to all compute in this project. Nothing heavy ever runs on the laptop.

From the repo root (the laptop's venv holds only `modal` and `ruff`):

    uv run --only-group local modal run infra/modal_app.py::test        # CPU unit tests
    uv run --only-group local modal run infra/modal_app.py::test_gpu    # GPU-marked tests on an L4
    uv run --only-group local modal run infra/modal_app.py::probe       # M0 hardware probe -> results/raw/
    uv run --only-group local modal run infra/modal_app.py::figures     # figures -> results/figures/
    uv run --only-group local modal run infra/modal_app.py::fetch       # download model weights (once)
    uv run --only-group local modal run infra/modal_app.py::m1          # M1 nanoserve measurements
    uv run --only-group local modal run infra/modal_app.py::m2          # M2 vLLM serving baselines

The container image is built in the cloud from pyproject.toml + uv.lock and cached after the first build.
Every function has a timeout, so a hung job can never eat the month's free credit.
"""

from __future__ import annotations

import json
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

# The serving image: vLLM pins its own PyTorch, so it gets its own environment (infra/serving.lock) instead of
# changing the research image that M0/M1 were measured with.
# It starts from NVIDIA's CUDA *devel* image: FlashInfer (used by vLLM, e.g. for sampling) compiles some
# kernels on first use and needs nvcc plus a host C++ compiler, which slim images don't have.
serving_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential")
    .uv_pip_install(requirements=[str(REPO / "infra" / "serving.lock")])
    .env(
        {
            "PYTHONPATH": f"{REMOTE}/src",
            "HF_HOME": f"{CACHE}/huggingface",
            "VLLM_CACHE_ROOT": f"{CACHE}/vllm",  # torch.compile and CUDA-graph artifacts survive between runs
            "FLASHINFER_WORKSPACE_BASE": f"{CACHE}/flashinfer",  # FlashInfer's compiled kernels, likewise
        }
    )
    .workdir(REMOTE)
    .add_local_dir(REPO / "src", f"{REMOTE}/src", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "benchmarks", f"{REMOTE}/benchmarks")
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
    from fastserve.engine.loader import model_dir

    path = model_dir(repo_id, download=True)
    hf_cache.commit()  # make the files visible to other containers
    return path


@app.function(gpu=GPU, cpu=2, memory=8192, timeout=30 * 60)
def hw_probe(config_text: str, config_path: str, git: dict) -> list[dict]:
    import yaml

    from fastserve.hw.probe import run_probe
    from fastserve.results import to_plain

    return to_plain(run_probe(yaml.safe_load(config_text), git=git, config_path=config_path))


@app.function(gpu=GPU, cpu=4, memory=32768, timeout=40 * 60, volumes={CACHE: hf_cache})
def m1_run(config_path: str, git: dict) -> list[dict]:
    import yaml

    from fastserve.engine.loader import model_dir
    from fastserve.experiments.m1 import run_m1
    from fastserve.results import to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    prompts = json.loads(Path(REMOTE, config["prompts"]).read_text(encoding="utf-8"))
    return to_plain(run_m1(config, prompts, model_dir(config["model"]), git=git, config_path=config_path))


@app.function(image=serving_image, gpu=GPU, cpu=4, memory=32768, timeout=30 * 60, volumes={CACHE: hf_cache})
def serving_smoke(model: str) -> dict:
    """Start vLLM, send one streaming request, and report versions and timings."""
    import json as _json
    import time
    import urllib.request
    from importlib.metadata import version

    from fastserve.engine.loader import model_dir
    from fastserve.serving.server import VLLMServer

    server = VLLMServer(model_dir(model), served_name=model)
    try:
        server.start()
    except (RuntimeError, TimeoutError) as err:
        return {"error": str(err).splitlines()[0], "log": server.log_path.read_text(errors="replace")}
    with server:  # already started; the context manager just guarantees stop()
        payload = {"model": model, "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}
        request = urllib.request.Request(
            f"{server.url}/v1/completions",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            body = _json.loads(response.read())
        hf_cache.commit()  # keep compile caches for next time
        return {
            "versions": {pkg: version(pkg) for pkg in ("vllm", "torch", "transformers", "lm_eval", "triton")},
            "startup_s": server.startup_s,
            "request_s": time.perf_counter() - start,
            "text": body["choices"][0]["text"],
            "log_tail": server.log_tail(25),
        }


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=120 * 60, volumes={CACHE: hf_cache})
def m2_run(config_path: str, git: dict) -> list[dict]:
    import yaml

    from fastserve.experiments.m2 import run_serving
    from fastserve.results import to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    workloads = yaml.safe_load(Path(REMOTE, config["workloads_file"]).read_text(encoding="utf-8"))
    records = run_serving(config, workloads, git=git, config_path=config_path)
    hf_cache.commit()  # keep vLLM/FlashInfer compile caches for the next run
    return to_plain(records)


@app.function(cpu=2, memory=4096, timeout=10 * 60)
def render_figures(milestone: str, raw: dict[str, list[dict]]) -> dict[str, bytes]:
    """Draw one milestone's figures from its raw records; returns {file name: bytes}."""
    import tempfile

    from fastserve.engine.config import ModelConfig
    from fastserve.viz import hw_figures, m1_figures

    with tempfile.TemporaryDirectory() as tmp:
        if milestone == "m0":
            written = hw_figures.make_all(raw["hw"], tmp)
        elif milestone == "m1":
            name = raw["m1"][0]["config"]["model"].split("/")[-1]
            cfg = ModelConfig.from_pretrained_json(
                Path(REMOTE, "benchmarks", "models", f"{name}.config.json")
            )
            written = m1_figures.make_all(raw["m1"], raw["hw"], cfg, tmp)
        else:
            raise ValueError(f"unknown milestone {milestone!r}")
        return {path.name: path.read_bytes() for path in written}


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
def m1(config: str = "benchmarks/configs/m1_nanoserve.yaml") -> None:
    from fastserve.results import append_jsonl, git_info

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    records = m1_run.remote(config, git)
    out = REPO / "results" / "raw" / "m1_nanoserve.jsonl"
    print(f"wrote {append_jsonl(out, records)} records to {out.relative_to(REPO)}")


@app.local_entrypoint()
def smoke(model: str = DEFAULT_MODEL) -> None:
    result = serving_smoke.remote(model)
    if "log" in result:  # startup failed: save the full vLLM log for diagnosis
        (REPO / "vllm_smoke.log").write_text(result["log"], encoding="utf-8")
        raise SystemExit(f"{result['error']} (full log in vllm_smoke.log)")
    print(json.dumps({k: v for k, v in result.items() if k != "log_tail"}, indent=2))
    print("--- vLLM log tail ---")
    print(result["log_tail"])


@app.local_entrypoint()
def m2(config: str = "benchmarks/configs/m2_baseline.yaml") -> None:
    from fastserve.results import append_jsonl, git_info

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    records = m2_run.remote(config, git)
    out = REPO / "results" / "raw" / "m2_serving.jsonl"
    print(f"wrote {append_jsonl(out, records)} records to {out.relative_to(REPO)}")


@app.local_entrypoint()
def figures(milestone: str = "all") -> None:
    from fastserve.results import latest_run, read_jsonl

    raw_dir = REPO / "results" / "raw"
    raw = {
        key: latest_run(read_jsonl(raw_dir / file))
        for key, file in (("hw", "hw_probe.jsonl"), ("m1", "m1_nanoserve.jsonl"))
        if (raw_dir / file).exists()
    }
    out = REPO / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ms in ["m0", "m1"] if milestone == "all" else [milestone]:
        if ms == "m1" and "m1" not in raw:
            continue
        for name, data in render_figures.remote(ms, raw).items():
            (out / name).write_bytes(data)
            print(f"wrote results/figures/{name}")
