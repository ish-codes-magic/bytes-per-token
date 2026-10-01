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
    if unequal:  # is each copy on the grid its stored scales define? (value / scale should be an integer)
        from safetensors import safe_open

        key = unequal[0]
        module = key.rsplit(".", 1)[0]
        folder = Path(f"{CHECKPOINT_DIR}/{name}")
        with safe_open(next(folder.glob("*.safetensors")), "pt") as f:
            scale = f.get_tensor(f"{module}.weight_scale").float()
        group = ours[key].shape[1] // scale.shape[1]
        grid = scale.repeat_interleave(group, dim=1)

        def off_grid(w: torch.Tensor) -> float:
            steps = w.float() / grid
            return (steps - steps.round()).abs().max().item()

        diff = ours[key].float() - library[key].float()
        diagnosis = {
            "tensor": key,
            "unequal_fraction": (diff != 0).float().mean().item(),
            "max_diff_in_steps": (diff.abs() / grid).max().item(),
            "ours_off_grid": off_grid(ours[key]),
            "library_off_grid": off_grid(library[key]),
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
    for key, file in (("m3", "m3_quant.jsonl"), ("m4", "m4_production.jsonl")):
        if (
            raw_dir / file
        ).exists():  # every run: tasks can be re-run, and the reports keep the newest result
            raw[key] = read_jsonl(raw_dir / file)
    out = REPO / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ms in ["m0", "m1", "m2", "m3", "m4"] if milestone == "all" else [milestone]:
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
