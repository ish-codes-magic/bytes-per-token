"""Result records: one JSON line per measurement, carrying everything needed to trust and reproduce it.

A record looks like:

    {"schema": 1, "experiment": "bandwidth", "run_id": "3f2a...", "timestamp": "2026-09-29T20:00:00+00:00",
     "metrics": {...}, "config": {...}, "git": {"commit": "...", "dirty": false}, "env": {...}}

`git` is collected where the repository lives (the laptop) and `env` where the measurement ran (a cloud
container). That split is why they are separate arguments to `make_record`.

Stdlib only: this module is also imported on the laptop, which has no ML packages installed.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Set by Modal inside its containers; recorded so every result can be traced to the machine that produced it.
_MODAL_ENV_VARS = ("MODAL_IMAGE_ID", "MODAL_REGION", "MODAL_CLOUD_PROVIDER")


def new_run_id() -> str:
    """A short random id shared by all records produced in one run."""
    return uuid.uuid4().hex[:12]


def git_info(repo_dir: str | Path = ".") -> dict[str, Any]:
    """Current commit, and whether the working tree has uncommitted changes (such results get flagged)."""

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True)
        return out.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_info() -> dict[str, Any]:
    """Hardware and software of the machine running the measurement. Call it where the work runs."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }
    info.update({name.lower(): value for name in _MODAL_ENV_VARS if (value := os.environ.get(name))})

    try:
        import torch
    except ImportError:
        return info
    # str(): torch.__version__ is a TorchVersion, and unpickling that on the laptop would need PyTorch.
    info["torch"] = str(torch.__version__)
    info["cuda_runtime"] = str(torch.version.cuda)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_sm_count"] = props.multi_processor_count
        info["gpu_memory_bytes"] = props.total_memory
        info["driver"] = _nvidia_driver_version()
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        pass
    info.update(_package_versions("vllm", "transformers", "flashinfer-python", "lm_eval"))
    return info


def _package_versions(*packages: str) -> dict[str, str]:
    """Versions of installed packages (read from metadata, so nothing gets imported)."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for package in packages:
        with contextlib.suppress(PackageNotFoundError):  # not installed in this image
            found[package.replace("-", "_")] = version(package)
    return found


def make_record(
    experiment: str,
    metrics: dict[str, Any],
    *,
    run_id: str,
    config: dict[str, Any] | None = None,
    git: dict[str, Any] | None = None,
    env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap one measurement with its metadata. `env` defaults to the current machine."""
    return {
        "schema": SCHEMA_VERSION,
        "experiment": experiment,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "config": config or {},
        "git": git or {},
        "env": env if env is not None else environment_info(),
    }


def to_plain(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A JSON round trip: guarantees only plain Python types remain.

    Cloud functions return their records through this. Results are pickled on the way back to the laptop,
    and a library type hiding inside (e.g. torch's TorchVersion, a str subclass) would need that library
    installed there just to unpickle.
    """
    return json.loads(json.dumps(records))


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a JSONL file (created if missing). Raw results are append-only. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_run(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records of the most recent run. Older runs stay in the file for variance studies."""
    if not records:
        return []
    last = max(records, key=lambda r: r["timestamp"])["run_id"]
    return [r for r in records if r["run_id"] == last]


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _nvidia_driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        return version.decode() if isinstance(version, bytes) else version
    except Exception:  # NVML missing, or blocked inside this container
        return None
