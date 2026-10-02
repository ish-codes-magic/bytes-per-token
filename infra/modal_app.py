"""Modal app: the single door to all compute in this project. Nothing heavy ever runs on the laptop.

From the repo root (the laptop's venv holds only `modal` and `ruff`):

    uv run --only-group local modal run infra/modal_app.py::test        # CPU unit tests
    uv run --only-group local modal run infra/modal_app.py::test_gpu    # GPU-marked tests on an L4
    uv run --only-group local modal run infra/modal_app.py::probe       # M0 hardware probe -> results/raw/
    uv run --only-group local modal run infra/modal_app.py::figures     # figures -> results/figures/
    uv run --only-group local modal run infra/modal_app.py::fetch       # download model weights (once)
    uv run --only-group local modal run infra/modal_app.py::m1          # M1 nanoserve measurements
    uv run --only-group local modal run infra/modal_app.py::m2          # M2 vLLM serving baselines
    uv run --only-group local modal run infra/modal_app.py::m2q         # M2 quality baselines
    uv run --only-group local modal run infra/modal_app.py::m3          # M3 quantization from scratch
    uv run --only-group local modal run infra/modal_app.py::m7          # M7 custom Triton kernels

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

# The library-quantization image: llm-compressor, whose GPTQ and AWQ check our reference implementations (M3).
# It resolves to the research image's PyTorch and transformers, but keeps its many extra packages to itself.
quant_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(requirements=[str(REPO / "infra" / "quant.lock")])
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": f"{CACHE}/huggingface"})
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
def m2_run(config_path: str, git: dict, only: list[str] | None = None) -> list[dict]:
    """Serving experiments from a config: every server, or only the named ones (one container each in M4)."""
    import yaml

    hf_cache.reload()  # a warm, reused container would otherwise miss checkpoints committed since it started

    from fastserve.experiments.m2 import run_serving
    from fastserve.results import to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    workloads = yaml.safe_load(Path(REMOTE, config["workloads_file"]).read_text(encoding="utf-8"))
    records = run_serving(config, workloads, git=git, config_path=config_path, only=only)
    hf_cache.commit()  # keep vLLM/FlashInfer compile caches for the next run
    return to_plain(records)


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=90 * 60, volumes={CACHE: hf_cache})
def m2_quality_task(task: str, model: str, config_path: str, git: dict) -> list[dict]:
    import yaml

    from fastserve.engine.loader import model_dir
    from fastserve.experiments.m2_quality import TASKS
    from fastserve.results import environment_info, make_record, new_run_id, to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    metrics = {"model": model, **TASKS[task](model_dir(model), config[task])}
    hf_cache.commit()  # datasets and compile caches, for next time
    record = make_record(
        f"quality_{task}",
        metrics,
        run_id=new_run_id(),
        config={"path": config_path, **config},
        git=git,
        env=environment_info(),
    )
    return to_plain([record])


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=60 * 60, volumes={CACHE: hf_cache})
def m2_offline_run(model: str, config_path: str, git: dict) -> list[dict]:
    """vLLM's engine alone on the throughput workload: separates engine capacity from serving overhead."""
    import yaml

    from fastserve.engine.loader import model_dir
    from fastserve.experiments.m2 import offline_throughput
    from fastserve.results import environment_info, make_record, new_run_id, to_plain
    from fastserve.serving.workloads import Workload

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    workloads = yaml.safe_load(Path(REMOTE, config["workloads_file"]).read_text(encoding="utf-8"))
    specs = Workload.from_config("throughput", workloads["throughput"]).requests()
    metrics = {"model": model, "workload": "throughput", **offline_throughput(model_dir(model), specs)}
    record = make_record(
        "offline_throughput",
        metrics,
        run_id=new_run_id(),
        config={"path": config_path, **config},
        git=git,
        env=environment_info(),
    )
    return to_plain([record])


@app.function(cpu=1, memory=1024, timeout=5 * 60)
def load_config(config_path: str) -> dict:
    """A YAML config, parsed in the cloud (the laptop has no PyYAML)."""
    import yaml

    return yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))


@app.function(gpu=GPU, cpu=4, memory=32768, timeout=150 * 60, volumes={CACHE: hf_cache})
def m3_task(task: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """One section of the M3 config (a set of quantized configurations, or an analysis) on one L4."""
    import yaml

    hf_cache.reload()  # see checkpoints committed by other containers since this one started

    from fastserve.experiments.m3 import run_task
    from fastserve.results import to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    return to_plain(run_task(task, config, run_id=run_id, git=git, config_path=config_path))


@app.function(gpu=GPU, cpu=4, memory=32768, timeout=120 * 60, volumes={CACHE: hf_cache})
def m5_task(task: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """One section of the M5 config's nanoserve tasks (KV policy KL, needle, statistics) on one L4."""
    import yaml

    hf_cache.reload()

    from fastserve.experiments.m5 import run_task
    from fastserve.results import to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    return to_plain(run_task(task, config, run_id=run_id, git=git, config_path=config_path))


@app.function(gpu=GPU, cpu=4, memory=32768, timeout=90 * 60, volumes={CACHE: hf_cache})
def m6_task(task: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """One section of the M6 config's nanoserve tasks (agreement, loop check, losslessness, timing)."""
    import yaml

    hf_cache.reload()

    from fastserve.experiments.m6 import run_task
    from fastserve.results import to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    records = run_task(task, config, run_id=run_id, git=git, config_path=config_path)
    hf_cache.commit()  # the prompt datasets, for the next container
    return to_plain(records)


def _m8_config(config_path: str) -> dict:
    """The M8 config with its plan expanded into the server list `run_serving` reads."""
    import yaml

    from fastserve.serving.ablation import expand

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    return {**config, "servers": expand(config)}


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=60 * 60, volumes={CACHE: hf_cache})
def m8_server(name: str, config_path: str, git: dict) -> list[dict]:
    """One server of the M8 plan through every workload it runs."""
    import yaml

    hf_cache.reload()

    from fastserve.experiments.m2 import run_serving
    from fastserve.results import to_plain

    config = _m8_config(config_path)
    workloads = yaml.safe_load(Path(REMOTE, config["workloads_file"]).read_text(encoding="utf-8"))
    records = run_serving(config, workloads, git=git, config_path=config_path, only=[name])
    hf_cache.commit()  # downloaded checkpoints and drafters, and compile caches, for the next container
    return to_plain(records)


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=30 * 60, volumes={CACHE: hf_cache})
def m8_smoke(name: str, config_path: str) -> dict:
    """Does this combination start at all? One short request, the server's own settings, nothing timed."""
    import json as _json
    import urllib.request

    hf_cache.reload()

    import urllib.error

    from fastserve.engine.loader import checkpoint_dir, model_dir
    from fastserve.experiments.m2 import _from_log, _speculative_args
    from fastserve.serving.server import VLLMServer

    entry = next(s for s in _m8_config(config_path)["servers"] if s["name"] == name)
    model = entry["model"]
    path = checkpoint_dir(entry["path"]) if entry.get("path") else model_dir(model)
    args = [*entry["args"], *_speculative_args(entry.get("speculative"))]
    server = VLLMServer(path, served_name=model, extra_args=args)
    try:
        server.start()
    except (RuntimeError, TimeoutError) as err:
        return {"name": name, "error": str(err).splitlines()[0], "log": server.log_tail(60)}
    with server:
        messages = [{"role": "user", "content": "Write a Python function that reverses a string."}]
        payload = {"model": model, "messages": messages, "max_tokens": 64, "temperature": 0}
        request = urllib.request.Request(
            f"{server.url}/v1/chat/completions",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = _json.loads(response.read())
        except urllib.error.HTTPError as err:  # the server is up but refused the request: say why
            reason = err.read().decode(errors="replace")[:600]
            return {"name": name, "error": f"HTTP {err.code}: {reason}", "log": server.log_tail(40)}
        with urllib.request.urlopen(f"{server.url}/metrics", timeout=30) as response:
            metrics = response.read().decode()
        spec = [line for line in metrics.splitlines() if "spec_decode" in line and not line.startswith("#")]
        hf_cache.commit()
        return {
            "name": name,
            "startup_s": server.startup_s,
            "text": body["choices"][0]["message"]["content"][:200],
            "settings": _from_log(server),
            "spec_counters": spec[:8],
        }


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=90 * 60, volumes={CACHE: hf_cache})
def m8_quality(kind: str, model: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """FP8 weights + FP8 KV in vLLM: WikiText-2 perplexity, the needle grid, or the task suite."""
    import yaml

    hf_cache.reload()

    from fastserve.engine.loader import checkpoint_dir
    from fastserve.experiments.m2_quality import needle_task, tasks_task, vllm_perplexity_task
    from fastserve.results import environment_info, make_record, to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))["quality"]
    cfg = {**config[kind], "kv_cache_dtype": config["kv_cache_dtype"]}
    path = checkpoint_dir(config["path"].format(model=model.split("/")[-1]))
    task = {"perplexity": vllm_perplexity_task, "needle": needle_task, "tasks": tasks_task}[kind]
    metrics = {"model": model, "label": config["label"], **task(path, cfg)}
    meta = {"path": config_path, kind: cfg}
    record = make_record(f"m8_{kind}", metrics, run_id=run_id, config=meta, git=git, env=environment_info())
    hf_cache.commit()
    return to_plain([record])


def _m7_task(task: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    import yaml

    hf_cache.reload()

    from fastserve.experiments.m7 import run_task
    from fastserve.results import to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    records = run_task(task, config, run_id=run_id, git=git, config_path=config_path, repo=REMOTE)
    hf_cache.commit()  # the evaluation text, and FlashInfer's compiled kernels, for the next container
    return to_plain(records)


@app.function(gpu=GPU, cpu=4, memory=32768, timeout=90 * 60, volumes={CACHE: hf_cache})
def m7_task(task: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """An M7 task that runs the real model in nanoserve (profile, end-to-end steps, quality)."""
    return _m7_task(task, config_path, run_id, git)


@app.function(image=serving_image, gpu=GPU, cpu=4, memory=32768, timeout=60 * 60, volumes={CACHE: hf_cache})
def m7_serving_task(task: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """An M7 microbenchmark next to what it competes with: vLLM's own ops and FlashInfer."""
    return _m7_task(task, config_path, run_id, git)


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=60 * 60, volumes={CACHE: hf_cache})
def m5_vllm_quality(kind: str, model: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """vLLM with an FP8 KV cache: the needle grid ("needle") or WikiText-2 perplexity ("perplexity")."""
    import yaml

    from fastserve.engine.loader import model_dir
    from fastserve.experiments.m2_quality import needle_task, vllm_perplexity_task
    from fastserve.results import environment_info, make_record, to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))["vllm_quality"]
    cfg = {**config[kind], "kv_cache_dtype": config["kv_cache_dtype"]}
    task = needle_task if kind == "needle" else vllm_perplexity_task
    metrics = {"model": model, "kv_cache_dtype": cfg["kv_cache_dtype"], **task(model_dir(model), cfg)}
    meta = {"path": config_path, kind: cfg}
    record = make_record(
        f"m5_vllm_{kind}", metrics, run_id=run_id, config=meta, git=git, env=environment_info()
    )
    return to_plain([record])


@app.function(image=quant_image, gpu=GPU, cpu=4, memory=32768, timeout=60 * 60, volumes={CACHE: hf_cache})
def m3_library(entry: dict, calibration: dict, config_path: str, run_id: str, git: dict) -> list[dict]:
    """llm-compressor quantizes the model; the dense result goes to the Volume, for nanoserve to score."""
    from fastserve.experiments.m3_library import LIBRARY_DIR, library_checkpoint
    from fastserve.results import environment_info, make_record, to_plain

    Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)
    out = f"{LIBRARY_DIR}/{entry['name']}.safetensors"
    info = library_checkpoint(entry["method"], entry["model"], calibration, out)
    hf_cache.commit()
    config = {"path": config_path, "entry": entry, "calibration": calibration}
    record = make_record(
        "m3_library_checkpoint", info, run_id=run_id, config=config, git=git, env=environment_info()
    )
    return to_plain([record])


@app.function(image=quant_image, gpu=GPU, cpu=4, memory=65536, timeout=150 * 60, volumes={CACHE: hf_cache})
def m4_checkpoint(fmt: str, model: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """llm-compressor makes one quantized checkpoint (compressed for vLLM, dense for nanoserve's KL)."""
    import yaml

    from fastserve.experiments.m4_checkpoints import make_checkpoint
    from fastserve.results import environment_info, make_record, to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    info = make_checkpoint(fmt, model, config["calibration"])
    hf_cache.commit()  # the other containers read the checkpoint from the Volume
    meta = {"path": config_path, "calibration": config["calibration"]}
    return to_plain(
        [make_record("m4_checkpoint", info, run_id=run_id, config=meta, git=git, env=environment_info())]
    )


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=90 * 60, volumes={CACHE: hf_cache})
def m4_suite(fmt: str, model: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """M2's lm-eval suite on one quantized checkpoint, served by vLLM with its real low-bit kernels."""
    import yaml

    hf_cache.reload()  # see checkpoints committed by other containers since this one started

    from fastserve.experiments.m2_quality import tasks_task
    from fastserve.experiments.m4_checkpoints import CHECKPOINT_DIR
    from fastserve.results import environment_info, make_record, to_plain

    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    path = f"{CHECKPOINT_DIR}/{model.split('/')[-1]}-{fmt}"
    metrics = {"model": model, "format": fmt, **tasks_task(path, config["suite"])}
    meta = {"path": config_path, "suite": config["suite"]}
    return to_plain(
        [make_record("m4_tasks", metrics, run_id=run_id, config=meta, git=git, env=environment_info())]
    )


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=60 * 60, volumes={CACHE: hf_cache})
def m4_fidelity(fmt: str, model: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """vLLM's own perplexity for one format, to compare with nanoserve's on the same rounded weights."""
    import yaml

    from fastserve.engine.loader import model_dir
    from fastserve.experiments.m2_quality import vllm_perplexity_task
    from fastserve.experiments.m4_checkpoints import CHECKPOINT_DIR
    from fastserve.results import environment_info, make_record, to_plain

    hf_cache.reload()
    config = yaml.safe_load(Path(REMOTE, config_path).read_text(encoding="utf-8"))
    path = model_dir(model) if fmt == "bf16" else f"{CHECKPOINT_DIR}/{model.split('/')[-1]}-{fmt}"
    metrics = {"model": model, "format": fmt, **vllm_perplexity_task(path, config["eval"])}
    meta = {"path": config_path, "eval": config["eval"]}
    record = make_record(
        "m4_vllm_perplexity", metrics, run_id=run_id, config=meta, git=git, env=environment_info()
    )
    return to_plain([record])


@app.function(cpu=4, memory=16384, timeout=30 * 60, volumes={CACHE: hf_cache})
def m4_decompress_check(fmt: str, model: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """Our compressed-tensors reader against llm-compressor's own dense export of the same checkpoint."""
    import torch
    from safetensors.torch import load_file

    from fastserve.experiments.m4_checkpoints import CHECKPOINT_DIR
    from fastserve.quant.compressed import load_dense
    from fastserve.results import environment_info, make_record, to_plain

    hf_cache.reload()
    name = f"{model.split('/')[-1]}-{fmt}"
    ours = load_dense(f"{CHECKPOINT_DIR}/{name}")
    library = load_file(f"{CHECKPOINT_DIR}/{name}.dense.safetensors")
    shared = sorted(set(ours) & set(library))
    unequal = [k for k in shared if not torch.equal(ours[k], library[k].to(ours[k].dtype))]
    max_diff = max(((ours[k].float() - library[k].float()).abs().max().item() for k in unequal), default=0.0)
    diagnosis = {}
    if unequal:  # how many weights differ, and by how many grid steps
        from safetensors import safe_open

        key = unequal[0]
        module = key.rsplit(".", 1)[0]
        with safe_open(next(Path(f"{CHECKPOINT_DIR}/{name}").glob("*.safetensors")), "pt") as f:
            scale = f.get_tensor(f"{module}.weight_scale").float()
        grid = scale.repeat_interleave(ours[key].shape[1] // scale.shape[1], dim=1)
        diff = ours[key].float() - library[key].float()
        diagnosis = {
            "tensor": key,
            "unequal_fraction": (diff != 0).float().mean().item(),
            "max_diff_in_steps": (diff.abs() / grid).max().item(),
        }
    metrics = {
        "model": model,
        "format": fmt,
        "tensors": len(shared),
        "only_ours": sorted(set(ours) - set(library)),
        "only_library": sorted(set(library) - set(ours)),
        "unequal_tensors": unequal,
        "max_abs_diff": max_diff,
        "diagnosis": diagnosis,
    }
    meta = {"path": config_path}
    record = make_record(
        "m4_decompress_check", metrics, run_id=run_id, config=meta, git=git, env=environment_info()
    )
    return to_plain([record])


@app.function(image=serving_image, gpu=GPU, cpu=8, memory=32768, timeout=60 * 60, volumes={CACHE: hf_cache})
def m4_gsm8k(fmt: str, model: str, config_path: str, run_id: str, git: dict) -> list[dict]:
    """GSM8K for one format with every answer logged, sorted into failure buckets."""
    from fastserve.engine.loader import model_dir
    from fastserve.experiments.m4_checkpoints import CHECKPOINT_DIR
    from fastserve.quality.tasks import gsm8k_failures
    from fastserve.results import environment_info, make_record, to_plain

    hf_cache.reload()
    path = model_dir(model) if fmt == "bf16" else f"{CHECKPOINT_DIR}/{model.split('/')[-1]}-{fmt}"
    metrics = {"model": model, "format": fmt, **gsm8k_failures(path)}
    meta = {"path": config_path}
    record = make_record("m4_gsm8k", metrics, run_id=run_id, config=meta, git=git, env=environment_info())
    return to_plain([record])


# Publishing needs a Hugging Face write token. The `publish` entrypoint attaches the Modal secret
# `huggingface-secret` (the name Modal's Hugging Face template gives it) at call time, so every other
# function runs without that secret existing.
HF_SECRET = "huggingface-secret"
HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def _hf_token() -> str:
    """The write token from whichever variable the secret defines. Never printed or returned to the laptop."""
    import os

    for name in HF_TOKEN_NAMES:
        if os.environ.get(name):
            return os.environ[name]
    raise RuntimeError(f"the Modal secret {HF_SECRET!r} defines none of {HF_TOKEN_NAMES}")


@app.function(cpu=1, memory=1024, timeout=5 * 60)
def hf_owner() -> str:
    """The Hugging Face account the token belongs to: the namespace the checkpoints are published under."""
    from huggingface_hub import HfApi

    return HfApi(token=_hf_token()).whoami()["name"]


@app.function(cpu=2, memory=8192, timeout=60 * 60, volumes={CACHE: hf_cache})
def hf_upload(folder: str, repo: str, card: str) -> dict:
    """Upload one checkpoint folder and its model card, then check each file on the Hub against our copy.

    A file counts as verified when the Hub's hash of it matches ours: SHA-256 for large (LFS) files, the git
    blob hash for small ones. Only fully verified checkpoints may be deleted from the Volume.
    """
    import hashlib

    from huggingface_hub import HfApi

    def sha256(data_or_path: bytes | Path) -> str:
        if isinstance(data_or_path, bytes):
            return hashlib.sha256(data_or_path).hexdigest()
        digest = hashlib.sha256()
        with open(data_or_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        return digest.hexdigest()

    def git_blob(data: bytes) -> str:
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

    hf_cache.reload()
    api = HfApi(token=_hf_token())
    api.create_repo(repo, repo_type="model", exist_ok=True)
    commit = api.upload_folder(folder_path=folder, repo_id=repo, commit_message="Upload checkpoint")
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        commit_message="Add the model card",
    )
    local = {p.name: p for p in sorted(Path(folder).iterdir()) if p.is_file()}
    remote = {info.path: info for info in api.get_paths_info(repo, [*local, "README.md"])}
    files = []
    for name, source in [*local.items(), ("README.md", card.encode("utf-8"))]:
        data = source if isinstance(source, bytes) else None
        size = len(data) if data is not None else source.stat().st_size
        info = remote.get(name)
        if info is None:
            ok = False
        elif getattr(info, "lfs", None):
            ok = info.lfs.sha256 == sha256(source) and info.size == size
        else:
            ok = info.blob_id == git_blob(data if data is not None else source.read_bytes())
        files.append({"path": name, "bytes": size, "verified": ok})
    return {
        "repo": repo,
        "url": f"https://huggingface.co/{repo}",
        "commit": commit.oid,
        "files": files,
        "verified": all(f["verified"] for f in files),
    }


@app.function(cpu=1, memory=1024, timeout=10 * 60, volumes={CACHE: hf_cache})
def remove_checkpoints(names: list[str]) -> list[str]:
    """Delete published checkpoints (and their dense exports) from the Volume; returns what was removed."""
    import shutil

    from fastserve.experiments.m4_checkpoints import CHECKPOINT_DIR

    hf_cache.reload()
    removed = []
    for name in names:
        for path in (Path(CHECKPOINT_DIR, name), Path(CHECKPOINT_DIR, f"{name}.dense.safetensors")):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            else:
                continue
            removed.append(str(path))
    hf_cache.commit()
    return removed


@app.function(cpu=2, memory=8192, timeout=20 * 60, volumes={CACHE: hf_cache})
def check_text_sources(model: str = DEFAULT_MODEL) -> dict:
    """Load a little of every calibration/evaluation source, so a missing dataset fails fast and cheaply."""
    from transformers import AutoTokenizer

    from fastserve.engine.loader import model_dir
    from fastserve.quality.text import SOURCES, calibration_ids, wikitext_eval_ids

    tokenizer = AutoTokenizer.from_pretrained(model_dir(model))
    found: dict = {"wikitext_test tokens": len(wikitext_eval_ids(tokenizer))}
    for name in SOURCES:
        if name == "wikitext_test":
            continue
        try:
            ids = calibration_ids(tokenizer, name, 4, 2048)
            found[name] = f"{tuple(ids.shape)}: {tokenizer.decode(ids[0, :24])!r}"
        except Exception as err:  # report every source, even after one fails
            found[name] = f"{type(err).__name__}: {err}"
    from fastserve.quality.prompts import TASKS, task_prompts

    for task in TASKS:  # M6's real prompts
        try:
            prompts = task_prompts(tokenizer, task, 8)
            lengths = [len(p) for p in prompts]
            found[f"prompts/{task}"] = (
                f"{min(lengths)}–{max(lengths)} tokens: {tokenizer.decode(prompts[0][-40:])!r}"
            )
        except Exception as err:
            found[f"prompts/{task}"] = f"{type(err).__name__}: {err}"
    hf_cache.commit()  # keep the downloaded shards for the real runs
    return found


@app.function(image=serving_image, cpu=2, memory=8192, timeout=15 * 60)
def serving_library_facts(topic: str = "lm_eval") -> str:
    """What the serving image's libraries do, read before relying on it (AGENTS.md §2.3).

    "lm_eval": how samples are logged and GSM8K is scored. "w8a8": what vLLM runs around an FP8/INT8 matmul.
    """
    import inspect
    import re
    from importlib.metadata import version
    from pathlib import Path

    if topic == "ops":  # M7: the norm and quantization ops our kernels compete with, and vLLM's fusion pass
        import vllm

        root = Path(vllm.__file__).parent
        source = (root / "_custom_ops.py").read_text(encoding="utf-8")
        lines = [f"vllm {version('vllm')}, triton {version('triton')}, torch {version('torch')}"]
        for name in re.findall(r"^def (\w*(?:norm|quant)\w*)\(", source, re.M):
            start = source.index(f"def {name}(")
            lines.append(source[start : start + 900].split("\n\n\n")[0])
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = re.finditer(
                r"^.*(RMSNormQuantFusion|enable_fusion|fuse_norm_quant|class .*Fusion.*Pass).*$", text, re.M
            )
            for match in list(hits)[:4]:
                lines.append(f"{path.relative_to(root)}: {match.group(0).strip()[:170]}")
        # Which quantization ops the norm+quant pass rewrites, and when the pass is switched on.
        lines.append(
            (root / "compilation/passes/fusion/rms_quant_fusion.py").read_text(encoding="utf-8")[:9000]
        )
        config = (root / "config/vllm.py").read_text(encoding="utf-8")
        for match in re.finditer(r"enable_norm_fusion|fuse_act_quant|optimization_level", config):
            lines.append(config[max(0, match.start() - 300) : match.start() + 500])
        # What the attention kernel can be compared with, and what Triton offers for rounding.
        import inspect

        import flashinfer
        from triton.language.extra import libdevice

        lines.append(f"flashinfer {version('flashinfer-python')}")
        for name in sorted(n for n in dir(flashinfer) if "decode" in n.lower()):
            obj = getattr(flashinfer, name)
            where = str(inspect.signature(obj)) if callable(obj) else type(obj).__name__
            lines.append(f"flashinfer.{name}{where}")
        lines.append("libdevice: " + ", ".join(n for n in dir(libdevice) if "int" in n or "round" in n))
        return "\n".join(lines)
    if topic == "spec":  # M6: vLLM's speculative-decoding config and metrics; drafters published for Qwen3
        import vllm
        from huggingface_hub import HfApi

        root = Path(vllm.__file__).parent
        source = (root / "config" / "speculative.py").read_text(encoding="utf-8")
        methods = source[source.index("SpeculativeMethod = ") :][:400]
        fields = source[source.index("class SpeculativeConfig") :][:7000]
        lines = [f"vllm {version('vllm')}", methods, fields]
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in list(re.finditer(r"^.*(spec_decode_\w+).*$", text, re.M))[:6]:
                lines.append(f"{path.relative_to(root)}: {match.group(0).strip()[:160]}")
        for query in ("Qwen3-1.7B eagle", "Qwen3 eagle3", "Qwen3-1.7B speculator"):
            found = [m.id for m in HfApi().list_models(search=query, limit=20)]
            lines.append(f"Hub search {query!r}: {found}")
        return "\n".join(lines)
    if topic == "kv":  # M5: KV-cache dtypes and scales, prefix-cache metrics, per-request cached tokens
        import re

        import vllm

        lines = [f"vllm {version('vllm')}"]
        patterns = [
            r"CacheDType\s*=",
            r"calculate_kv_scales",
            r"prefix_cache_(hits|queries)",
            r"enable_prompt_tokens_details",
            r"Maximum concurrency for",
            r"cached_tokens",
            r"fp8.*kv cache|kv cache.*fp8",
            r"VLLM_ATTENTION_BACKEND",
            r"--attention-backend|attention_backend:",
            r"Using \S+ backend",
        ]
        root = Path(vllm.__file__).parent
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                for match in list(re.finditer(rf"^.*({pattern}).*$", text, re.M | re.I))[:3]:
                    lines.append(f"{path.relative_to(root)}: {match.group(0).strip()[:160]}")
        return "\n".join(lines[:400])
    if topic == "w8a8":  # the kernels vLLM's startup log named for FP8 and INT8 W8A8
        import vllm

        lines = [f"vllm {version('vllm')}"]
        names = ("CutlassFP8ScaledMMLinearKernel", "CutlassInt8ScaledMMLinearKernel")
        for path in sorted(Path(vllm.__file__).parent.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for name in names:
                at = text.find(f"class {name}")
                if at >= 0:
                    body = text[at:]
                    apply = body.find("def apply_weights")
                    lines.append(f"{path}: {name}")
                    lines.append(body[apply : apply + 2500] if apply >= 0 else body[:2500])
        return "\n".join(lines)

    import lm_eval
    import lm_eval.evaluator as evaluator

    lines = [f"lm_eval {version('lm_eval')}"]
    source = inspect.getsource(evaluator)
    for match in re.finditer(
        r"^.*(filtered_resps|\"filter\"|\"doc_id\"|\"resps\"|\"target\").*$", source, re.M
    ):
        lines.append(match.group(0).rstrip())
    tasks = Path(lm_eval.__file__).parent / "tasks" / "gsm8k"
    lines.append((tasks / "gsm8k.yaml").read_text(encoding="utf-8"))
    return "\n".join(lines)


@app.function(image=quant_image, cpu=2, memory=8192, timeout=15 * 60, volumes={CACHE: hf_cache})
def quant_library_facts(topic: str = "m3") -> str:
    """What the installed llm-compressor does, read before relying on it (AGENTS.md §2.3).

    "m3": GPTQ/AWQ defaults and the grid formula. "m4": the presets, SmoothQuant and the oneshot/save APIs.
    """
    import importlib
    import inspect
    from importlib.metadata import version

    from compressed_tensors.quantization import preset_name_to_scheme
    from compressed_tensors.quantization.utils import helpers

    lines = [f"llmcompressor {version('llmcompressor')}, compressed-tensors {version('compressed-tensors')}"]
    if topic == "compressed":  # the on-disk format of M4's checkpoints, to decompress them ourselves
        import json

        from safetensors import safe_open

        hf_cache.reload()
        for name in ("Qwen3-0.6B-fp8", "Qwen3-0.6B-int8", "Qwen3-0.6B-gptq"):
            folder = Path(CACHE, "m4", name)
            lines.append(f"== {name}: {sorted(p.name for p in folder.iterdir())}")
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            lines.append(json.dumps(config.get("quantization_config"), indent=1)[:2500])
            for shard in sorted(folder.glob("*.safetensors")):
                with safe_open(shard, "pt") as f:
                    for key in sorted(f.keys()):
                        if (
                            "layers.0.mlp.down_proj" in key
                            or "layers.0.self_attn.q_proj" in key
                            or "embed" in key
                        ):
                            t = f.get_slice(key)
                            lines.append(f"  {key}: {t.get_dtype()} {t.get_shape()}")
        import compressed_tensors

        for path in sorted(Path(compressed_tensors.__file__).parent.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for name in ("def pack_to_int32", "def unpack_from_int32"):
                at = text.find(name)
                if at >= 0:
                    lines.append(f"{path}:")
                    lines.append(text[at : at + 2500])
        return "\n".join(lines)
    if topic == "m4":
        for preset in ("FP8_DYNAMIC", "W8A8", "W4A16"):
            lines.append(f"{preset}: {preset_name_to_scheme(preset, ['Linear'])}")
        for module in ("llmcompressor.modifiers.smoothquant", "llmcompressor.modifiers.transform"):
            try:
                names = [n for n in dir(importlib.import_module(module)) if "Modifier" in n]
                lines.append(f"{module}: {names}")
            except ImportError as err:
                lines.append(f"{module}: {err}")
        from llmcompressor import oneshot

        lines.append(f"oneshot{inspect.signature(oneshot)}")
        from llmcompressor.transformers.compression.compressed_tensors_utils import modify_save_pretrained

        lines.append(inspect.getsource(modify_save_pretrained)[:3000])
        return "\n".join(lines)
    for module, name in (
        ("llmcompressor.modifiers.gptq", "GPTQModifier"),
        ("llmcompressor.modifiers.transform", "AWQModifier"),
        ("llmcompressor.modifiers.gptq.gptq_quantize", "quantize_weight"),
    ):
        try:
            obj = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as err:
            lines.append(f"{module}.{name}: {err}")
            continue
        lines.append(f"{module}.{name}: {type(obj).__name__} defined in {getattr(obj, '__module__', '?')}")
        fields = getattr(obj, "model_fields", None)
        if fields:
            lines += [f"  {field} = {info.default!r}" for field, info in fields.items()]
        else:
            lines.append(inspect.getsource(obj)[:3000])
    lines.append(f"W4A16 preset: {preset_name_to_scheme('W4A16', ['Linear'])}")
    lines.append(inspect.getsource(helpers.calculate_qparams))
    awq = importlib.import_module("llmcompressor.modifiers.transform.awq.base")
    for attr in ("_compute_best_scale", "_run_samples"):  # what AWQ's loss compares
        method = getattr(awq.AWQModifier, attr, None)
        lines.append(inspect.getsource(method)[:5000] if method else f"no {attr}")
    try:  # which layers AWQ scales together for Qwen3
        mappings = importlib.import_module("llmcompressor.modifiers.transform.awq.mappings")
        lines.append(inspect.getsource(mappings)[:5000])
    except (ImportError, OSError) as err:
        lines.append(f"AWQ mappings: {err}")
    return "\n".join(lines)


@app.function(cpu=2, memory=4096, timeout=10 * 60)
def render_figures(milestone: str, raw: dict[str, list[dict]]) -> dict[str, bytes]:
    """Draw one milestone's figures from its raw records; returns {file name: bytes}."""
    import tempfile

    from fastserve.engine.config import ModelConfig
    from fastserve.viz import hw_figures, m1_figures, m2_figures

    with tempfile.TemporaryDirectory() as tmp:
        if milestone == "m0":
            written = hw_figures.make_all(raw["hw"], tmp)
        elif milestone == "m1":
            name = raw["m1"][0]["config"]["model"].split("/")[-1]
            cfg = ModelConfig.from_pretrained_json(
                Path(REMOTE, "benchmarks", "models", f"{name}.config.json")
            )
            written = m1_figures.make_all(raw["m1"], raw["hw"], cfg, tmp)
        elif milestone == "m2":
            written = m2_figures.make_all(raw["m2"], tmp)
            if raw.get("m2q"):
                written += m2_figures.make_quality(raw["m2q"], tmp)
            if raw.get("m2sat"):
                from fastserve.hw.analysis import measured_bandwidth

                configs = {
                    model: ModelConfig.from_pretrained_json(
                        Path(REMOTE, "benchmarks", "models", f"{model.split('/')[-1]}.config.json")
                    )
                    for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
                }
                bandwidth = measured_bandwidth(raw["hw"])
                written += m2_figures.make_saturation(raw["m2sat"], configs, bandwidth, tmp)
        elif milestone == "m3":
            from fastserve.viz import m3_figures

            written = m3_figures.make_all(raw["m3"], tmp)
        elif milestone == "m4":
            from fastserve.hw.analysis import measured_bandwidth, measured_peak_flops
            from fastserve.viz import m4_figures

            configs = {
                model: ModelConfig.from_pretrained_json(
                    Path(REMOTE, "benchmarks", "models", f"{model.split('/')[-1]}.config.json")
                )
                for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
            }
            peaks = measured_peak_flops(raw["hw"])
            hw = {"bandwidth": measured_bandwidth(raw["hw"]), "bf16": peaks["bf16"], "fp8": peaks["fp8"]}
            written = m4_figures.make_all(raw["m4"], configs, hw, tmp)
        elif milestone == "m5":
            import yaml

            from fastserve.viz import m5_figures

            config = yaml.safe_load(
                Path(REMOTE, "benchmarks", "configs", "m5_kv.yaml").read_text(encoding="utf-8")
            )
            workloads = yaml.safe_load(Path(REMOTE, config["workloads_file"]).read_text(encoding="utf-8"))
            cfg = ModelConfig.from_pretrained_json(
                Path(REMOTE, "benchmarks", "models", "Qwen3-0.6B.config.json")
            )
            written = m5_figures.make_all(raw["m5"], config["policies"], cfg, workloads, tmp)
        elif milestone == "m6":
            from fastserve.viz import m6_figures

            written = m6_figures.make_all(raw["m6"], tmp)
        elif milestone == "m7":
            import yaml

            from fastserve.hw.analysis import measured_bandwidth
            from fastserve.viz import m7_figures

            config = yaml.safe_load(
                Path(REMOTE, "benchmarks", "configs", "m5_kv.yaml").read_text(encoding="utf-8")
            )  # the price of an L4 hour, as in M5's cost tables
            cfg = ModelConfig.from_pretrained_json(
                Path(REMOTE, "benchmarks", "models", "Qwen3-0.6B.config.json")
            )
            bandwidth = measured_bandwidth(raw["hw"])
            written = m7_figures.make_all(raw["m7"], bandwidth, cfg, config["dollars_per_hour"], tmp)
        elif milestone == "m8":
            import yaml

            from fastserve.report import m8 as m8_report
            from fastserve.viz import m8_figures

            config = yaml.safe_load(
                Path(REMOTE, "benchmarks", "configs", "m8_ablation.yaml").read_text(encoding="utf-8")
            )
            frozen = json.loads(
                Path(REMOTE, "benchmarks", "predictions", "m8_model.json").read_text(encoding="utf-8")
            )
            configs = {
                model: ModelConfig.from_pretrained_json(
                    Path(REMOTE, "benchmarks", "models", f"{model.split('/')[-1]}.config.json")
                )
                for model in (m8_report.SMALL, m8_report.LARGE)
            }
            perplexity = {
                model: m8_report.perplexities(raw["m8"], raw["m5"], raw["m4"], model) for model in configs
            }
            projected = {}
            for model in configs:
                found = m8_report.kernel_projection(raw["m8"], raw["m7"], configs, model)
                if found:
                    projected[model] = found["tok_s"]
            written = m8_figures.make_all(
                raw["m8"],
                frozen,
                configs,
                m8_report.hardware(raw["hw"]),
                config["dollars_per_hour"],
                perplexity,
                config["techniques"]["s"]["head"],
                projected,
                tmp,
            )
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


@app.function(cpu=1, memory=1024, timeout=5 * 60, volumes={CACHE: hf_cache})
def read_model_config(repo_id: str) -> str:
    from fastserve.engine.loader import model_dir

    return Path(model_dir(repo_id), "config.json").read_text(encoding="utf-8")


@app.local_entrypoint()
def fetch(model: str = DEFAULT_MODEL) -> None:
    """Download a model into the Volume, and save its config.json under benchmarks/models/ (it's small)."""
    print(f"{model} is at {download_model.remote(model)} on the {hf_cache.name or 'cache'} volume")
    out = REPO / "benchmarks" / "models" / f"{model.split('/')[-1]}.config.json"
    out.write_text(read_model_config.remote(model), encoding="utf-8", newline="\n")
    print(f"saved {out.relative_to(REPO)}")


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
def m2(config: str = "benchmarks/configs/m2_baseline.yaml", out: str = "m2_serving.jsonl") -> None:
    from fastserve.results import append_jsonl, git_info

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    records = m2_run.remote(config, git)
    out = REPO / "results" / "raw" / out
    print(f"wrote {append_jsonl(out, records)} records to {out.relative_to(REPO)}")


@app.local_entrypoint()
def m2offline(config: str = "benchmarks/configs/m2_baseline.yaml") -> None:
    from fastserve.results import append_jsonl, git_info

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    out = REPO / "results" / "raw" / "m2_offline.jsonl"
    calls = [m2_offline_run.spawn(model, config, git) for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")]
    for call in calls:
        records = call.get()
        m = records[0]["metrics"]
        print(f"wrote {append_jsonl(out, records)}: {m['model']} {m['output_throughput']:,.0f} tok/s offline")


@app.local_entrypoint()
def m2q(config: str = "benchmarks/configs/m2_quality.yaml", tasks: str = "perplexity,needle,tasks") -> None:
    """Quality baselines: every (task, model) pair runs in parallel in its own container."""
    from fastserve.results import append_jsonl, git_info

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    models = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
    calls = [m2_quality_task.spawn(task, model, config, git) for task in tasks.split(",") for model in models]
    out = REPO / "results" / "raw" / "m2_quality.jsonl"
    failed = 0
    for call in calls:  # collect each separately: one failure mustn't discard the others' results
        try:
            records = call.get()
        except Exception as err:  # the remote traceback is already in the log above
            failed += 1
            print(f"FAILED: {type(err).__name__}: {str(err).splitlines()[0][:200]}")
            continue
        first = records[0]
        name = f"{first['experiment']} {first['metrics']['model']}"
        print(f"wrote {append_jsonl(out, records)} record(s): {name}")
    if failed:
        raise SystemExit(f"{failed} quality task(s) failed")


@app.local_entrypoint()
def figures(milestone: str = "all") -> None:
    from fastserve.results import latest_run, read_jsonl

    raw_dir = REPO / "results" / "raw"
    raw = {
        key: latest_run(read_jsonl(raw_dir / file))
        for key, file in (
            ("hw", "hw_probe.jsonl"),
            ("m1", "m1_nanoserve.jsonl"),
            ("m2", "m2_serving.jsonl"),
            ("m2sat", "m2_saturation.jsonl"),
        )
        if (raw_dir / file).exists()
    }
    quality = raw_dir / "m2_quality.jsonl"
    if quality.exists():  # gathered from one container per (task, model): keep the newest of each
        from fastserve.report.m2 import newest_per_model

        raw["m2q"] = newest_per_model(read_jsonl(quality))
    for key, file in (
        ("m3", "m3_quant.jsonl"),
        ("m4", "m4_production.jsonl"),
        ("m5", "m5_kv.jsonl"),
        ("m6", "m6_spec.jsonl"),
        ("m7", "m7_kernels.jsonl"),
        ("m8", "m8_ablation.jsonl"),
    ):
        if (
            raw_dir / file
        ).exists():  # every run: tasks can be re-run, and the reports keep the newest result
            raw[key] = read_jsonl(raw_dir / file)
    out = REPO / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ms in ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"] if milestone == "all" else [milestone]:
        if ms != "m0" and ms not in raw:
            continue
        for name, data in render_figures.remote(ms, raw).items():
            (out / name).write_bytes(data)
            print(f"wrote results/figures/{name}")


@app.local_entrypoint()
def m3(config: str = "benchmarks/configs/m3_quant.yaml", tasks: str = "") -> None:
    """M3: llm-compressor's checkpoints and every task of the config, each in its own container.

    `--tasks grids,gptq` runs a subset. "library" makes llm-compressor's checkpoints; "library_eval" scores
    them, and starts as soon as they exist.
    """
    from fastserve.results import append_jsonl, git_info, new_run_id

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    cfg = load_config.remote(config)
    wanted = tasks.split(",") if tasks else ["library", *cfg["tasks"]]
    run_id, out = new_run_id(), REPO / "results" / "raw" / "m3_quant.jsonl"
    library = []
    if "library" in wanted:  # llm-compressor's checkpoints, made while the other tasks run
        library = [m3_library.spawn(e, cfg["calibration"], config, run_id, git) for e in cfg["library"]]
    calls = {t: m3_task.spawn(t, config, run_id, git) for t in wanted if t not in ("library", "library_eval")}
    for call in library:
        records = call.get()
        print(f"wrote {append_jsonl(out, records)} library checkpoint record: {records[0]['metrics']}")
    if "library_eval" in wanted:
        calls["library_eval"] = m3_task.spawn("library_eval", config, run_id, git)
    for task, call in calls.items():
        try:
            print(f"{task}: wrote {append_jsonl(out, call.get())} records to {out.relative_to(REPO)}")
        except Exception as err:  # keep what the other tasks produced
            print(f"{task} FAILED: {type(err).__name__}: {err}")


@app.local_entrypoint()
def m4(
    config: str = "benchmarks/configs/m4_production.yaml",
    steps: str = "checkpoints,serving,suite,kl,fidelity,gsm8k",
    formats: str = "",
    models: str = "",
) -> None:
    """M4: quantized checkpoints, then their servers, task scores and KL, each in its own container.

    BF16 servers start at once. Each checkpoint's server and task scores start as soon as it lands, and a
    model's KL run once all its formats exist. Without "checkpoints" in `steps`, they must exist already.
    `formats` and `models` (comma-separated) re-run a subset, e.g. one format that failed.
    """
    from fastserve.results import append_jsonl, git_info, new_run_id

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    cfg = load_config.remote(config)
    wanted = set(steps.split(","))
    run_id, out = new_run_id(), REPO / "results" / "raw" / "m4_production.jsonl"
    formats_given = bool(formats)
    models = models.split(",") if models else cfg["checkpoints"]["models"]
    formats = formats.split(",") if formats else cfg["checkpoints"]["formats"]
    server = {(s["model"], s["label"]): s["name"] for s in cfg["servers"]}
    kl_task = {
        spec.get("model", cfg["model"]) if isinstance(spec, dict) else cfg["model"]: name
        for name, spec in cfg["tasks"].items()
    }
    calls: list[tuple[str, object]] = []

    def dependents(model: str, fmt: str) -> None:
        if "serving" in wanted:
            calls.append((f"serving {server[model, fmt]}", m2_run.spawn(config, git, [server[model, fmt]])))
        if "suite" in wanted:
            calls.append((f"suite {model} {fmt}", m4_suite.spawn(fmt, model, config, run_id, git)))
        if "fidelity" in wanted:
            calls.append((f"fidelity {model} {fmt}", m4_fidelity.spawn(fmt, model, config, run_id, git)))
        if "gsm8k" in wanted:
            calls.append((f"gsm8k {model} {fmt}", m4_gsm8k.spawn(fmt, model, config, run_id, git)))
        if "decompress" in wanted:  # needs the dense exports, so not a default step
            check = m4_decompress_check.spawn(fmt, model, config, run_id, git)
            calls.append((f"decompress {model} {fmt}", check))

    for model in [] if formats_given else models:  # BF16 needs no checkpoint
        if "serving" in wanted:
            calls.append(
                (f"serving {server[model, 'bf16']}", m2_run.spawn(config, git, [server[model, "bf16"]]))
            )
        if "fidelity" in wanted:
            calls.append((f"fidelity {model} bf16", m4_fidelity.spawn("bf16", model, config, run_id, git)))
        if "gsm8k" in wanted:
            calls.append((f"gsm8k {model} bf16", m4_gsm8k.spawn("bf16", model, config, run_id, git)))
    if "checkpoints" in wanted:
        made = {(m, f): m4_checkpoint.spawn(f, m, config, run_id, git) for m in models for f in formats}
        ready = dict.fromkeys(models, 0)
        for (model, fmt), call in made.items():
            try:
                records = call.get()
            except Exception as err:  # its dependents can't run; everything else still can
                print(f"checkpoint {model} {fmt} FAILED: {type(err).__name__}: {err}")
                continue
            print(f"checkpoint {model} {fmt}: {records[0]['metrics']} ({append_jsonl(out, records)} record)")
            dependents(model, fmt)
            ready[model] += 1
            if "kl" in wanted and ready[model] == len(formats):
                calls.append((f"kl {model}", m3_task.spawn(kl_task[model], config, run_id, git)))
    else:
        for model in models:
            for fmt in formats:
                dependents(model, fmt)
            if "kl" in wanted:
                calls.append((f"kl {model}", m3_task.spawn(kl_task[model], config, run_id, git)))
    for what, call in calls:
        try:
            print(f"{what}: wrote {append_jsonl(out, call.get())} records to {out.relative_to(REPO)}")
        except Exception as err:
            print(f"{what} FAILED: {type(err).__name__}: {err}")


@app.local_entrypoint()
def publish(formats: str = "", models: str = "", delete_local: bool = False) -> None:
    """Publish M4's checkpoints on the Hugging Face Hub with generated model cards (also kept in
    results/model_cards/). With --delete-local, checkpoints whose every file verified are then removed from
    the Volume: the Hub copy is the one that stays, and nanoserve reads it directly (quant/compressed.py).
    """
    from fastserve.report.m4 import FORMATS, LARGE, SMALL
    from fastserve.report.model_card import model_card as card_for
    from fastserve.report.model_card import repo_name
    from fastserve.results import append_jsonl, git_info, make_record, new_run_id, read_jsonl

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; the cards cite a commit that doesn't hold them")
    raw = REPO / "results" / "raw"
    m4_records, m2_quality = read_jsonl(raw / "m4_production.jsonl"), read_jsonl(raw / "m2_quality.jsonl")
    secret = [modal.Secret.from_name(HF_SECRET)]
    owner = hf_owner.with_options(secrets=secret).remote()
    uploader = hf_upload.with_options(secrets=secret)
    cards = REPO / "results" / "model_cards"
    cards.mkdir(parents=True, exist_ok=True)
    calls = []
    for model in models.split(",") if models else (SMALL, LARGE):
        for fmt in formats.split(",") if formats else [f for f in FORMATS if f != "bf16"]:
            repo = f"{owner}/{repo_name(model, fmt)}"
            card = card_for(m4_records, m2_quality, model, fmt, commit=git["commit"], repo=repo)
            (cards / f"{repo_name(model, fmt)}.md").write_text(card, encoding="utf-8", newline="\n")
            folder = f"/cache/m4/{model.split('/')[-1]}-{fmt}"
            calls.append((model, fmt, folder, uploader.spawn(folder, repo, card)))
    run_id, published = new_run_id(), []
    for model, fmt, folder, call in calls:
        try:
            result = call.get()
        except Exception as err:
            print(f"publish {model} {fmt} FAILED: {type(err).__name__}: {err}")
            continue
        record = make_record("m4_publish", {"model": model, "format": fmt, **result}, run_id=run_id, git=git)
        append_jsonl(raw / "m4_production.jsonl", [record])
        print(f"{result['url']}: {'verified' if result['verified'] else 'NOT verified'}")
        if result["verified"]:
            published.append(Path(folder).name)
    if delete_local and published:
        for path in remove_checkpoints.remote(published):
            print(f"removed {path}")


@app.local_entrypoint()
def m5(
    config: str = "benchmarks/configs/m5_kv.yaml", steps: str = "tasks,serving,quality", only: str = ""
) -> None:
    """M5: nanoserve's KV-policy tasks, vLLM's servers, and vLLM's FP8-KV quality, all in parallel containers.

    `--only` (comma-separated) picks task names and server names, e.g. `--only kl_small,0.6b-fp8kv`.
    """
    from fastserve.results import append_jsonl, git_info, new_run_id

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    cfg = load_config.remote(config)
    wanted, picked = set(steps.split(",")), set(only.split(",")) if only else None
    run_id, out = new_run_id(), REPO / "results" / "raw" / "m5_kv.jsonl"
    calls: list[tuple[str, object]] = []
    if "tasks" in wanted:
        for task in cfg["tasks"]:
            if picked is None or task in picked:
                calls.append((task, m5_task.spawn(task, config, run_id, git)))
    if "serving" in wanted:
        for server in cfg["servers"]:
            if picked is None or server["name"] in picked:
                calls.append((f"serving {server['name']}", m2_run.spawn(config, git, [server["name"]])))
    if "quality" in wanted:
        for model in cfg["vllm_quality"]["models"]:
            for kind in ("needle", "perplexity"):
                if picked is None or f"vllm-{kind}" in picked:
                    calls.append(
                        (f"vllm {kind} {model}", m5_vllm_quality.spawn(kind, model, config, run_id, git))
                    )
    for what, call in calls:
        try:
            print(
                f"{what}: wrote {append_jsonl(out, call.get())} records to {out.relative_to(REPO)}",
                flush=True,
            )
        except Exception as err:  # keep what the other containers produced
            print(f"{what} FAILED: {type(err).__name__}: {err}", flush=True)


@app.local_entrypoint()
def m6(config: str = "benchmarks/configs/m6_spec.yaml", steps: str = "tasks,serving", only: str = "") -> None:
    """M6: nanoserve's speculative-decoding tasks and vLLM's servers, each in its own container.

    `--only` (comma-separated) picks task names and server names, e.g. `--only loop,bf16-draft-k3`.
    """
    from fastserve.results import append_jsonl, git_info, new_run_id

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    cfg = load_config.remote(config)
    wanted, picked = set(steps.split(",")), set(only.split(",")) if only else None
    run_id, out = new_run_id(), REPO / "results" / "raw" / "m6_spec.jsonl"
    calls: list[tuple[str, object]] = []
    if "tasks" in wanted:
        for task in cfg["tasks"]:
            if picked is None or task in picked:
                calls.append((task, m6_task.spawn(task, config, run_id, git)))
    if "serving" in wanted:
        for server in cfg["servers"]:
            if picked is None or server["name"] in picked:
                calls.append((f"serving {server['name']}", m2_run.spawn(config, git, [server["name"]])))
    for what, call in calls:
        try:
            print(
                f"{what}: wrote {append_jsonl(out, call.get())} records to {out.relative_to(REPO)}",
                flush=True,
            )
        except Exception as err:  # keep what the other containers produced
            print(f"{what} FAILED: {type(err).__name__}: {err}", flush=True)


@app.local_entrypoint()
def m7(config: str = "benchmarks/configs/m7_kernels.yaml", only: str = "") -> None:
    """M7: the kernels' profile, microbenchmarks, nanoserve runs and quality, each task in its own container.

    `--only` (comma-separated) picks tasks, e.g. `--only profile,ops`. A task's `image` decides where it
    runs: `research` (nanoserve, as M1–M6) or `serving` (next to vLLM's ops and FlashInfer).
    """
    from fastserve.results import append_jsonl, git_info, new_run_id

    git = git_info(REPO)
    if git["dirty"]:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    cfg = load_config.remote(config)
    picked = set(only.split(",")) if only else None
    run_id, out = new_run_id(), REPO / "results" / "raw" / "m7_kernels.jsonl"
    calls: list[tuple[str, object]] = []
    for task, spec in cfg["tasks"].items():
        if picked is None or task in picked:
            fn = m7_serving_task if spec["image"] == "serving" else m7_task
            calls.append((task, fn.spawn(task, config, run_id, git)))
    for what, call in calls:
        try:
            print(
                f"{what}: wrote {append_jsonl(out, call.get())} records to {out.relative_to(REPO)}",
                flush=True,
            )
        except Exception as err:  # keep what the other containers produced
            print(f"{what} FAILED: {type(err).__name__}: {err}", flush=True)


@app.function(cpu=1, memory=1024, timeout=5 * 60)
def m8_plan(config_path: str) -> list[dict]:
    """The plan's servers (name, label, model), expanded where the config and PyYAML live."""
    return [{k: s[k] for k in ("name", "label", "model")} for s in _m8_config(config_path)["servers"]]


@app.local_entrypoint()
def m8(
    config: str = "benchmarks/configs/m8_ablation.yaml",
    only: str = "",
    check: bool = False,
    quality: str = "",
) -> None:
    """M8: every server of the ablation plan, one container each, results appended as they finish.

    `--only 1.7b-base,1.7b-wkps` picks servers by name. `--check` only verifies that the picked combinations
    start and answer one request (nothing is timed or recorded). `--quality perplexity,needle,tasks` runs
    the quality tasks for FP8 weights + FP8 KV instead of the servers.
    """
    from fastserve.results import append_jsonl, git_info, make_record, new_run_id

    git = git_info(REPO)
    if git["dirty"] and not check:
        print("warning: uncommitted changes; these results will be flagged as dirty")
    picked = set(only.split(",")) if only else None
    names = [s["name"] for s in m8_plan.remote(config) if picked is None or s["name"] in picked]
    if check:
        for result in m8_smoke.map(names, kwargs={"config_path": config}):
            print(json.dumps(result, indent=2)[:3000], flush=True)
        return
    out = REPO / "results" / "raw" / "m8_ablation.jsonl"
    if quality:
        run_id, models = new_run_id(), load_config.remote(config)["quality"]["models"]
        kinds = quality.split(",")
        calls = [
            (f"{kind} {model}", m8_quality.spawn(kind, model, config, run_id, git))
            for model in models
            for kind in kinds
        ]
    else:
        calls = [(name, m8_server.spawn(name, config, git)) for name in names]
    for name, call in calls:
        try:
            print(
                f"{name}: wrote {append_jsonl(out, call.get())} records to {out.relative_to(REPO)}",
                flush=True,
            )
        except Exception as err:  # keep what the other containers produced, and record the failure
            message = f"{type(err).__name__}: {str(err)[:1500]}"
            failure = make_record(
                "m8_failure", {"name": name, "error": message}, run_id=new_run_id(), git=git
            )
            append_jsonl(out, [failure])
            print(f"{name} FAILED: {message[:600]}", flush=True)
