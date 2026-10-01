"""Run `vllm serve` as a subprocess and wait until it answers."""

from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path


class VLLMServer:
    """Context manager: starts an OpenAI-compatible vLLM server on localhost and stops it on exit.

    Everything vLLM prints goes to `log_path`, so a failed start can be diagnosed from the log tail.
    """

    def __init__(
        self,
        model_path: str,
        *,
        served_name: str,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        port: int = 8000,
        log_path: str | Path = "/tmp/vllm.log",
        startup_timeout_s: float = 900,
    ):
        self.url = f"http://127.0.0.1:{port}"
        self.command = [
            "vllm",
            "serve",
            model_path,
            "--served-model-name",
            served_name,
            "--port",
            str(port),
            *(extra_args or []),
        ]
        self.env = env or {}  # extra environment variables, e.g. VLLM_ATTENTION_BACKEND
        self.log_path, self.startup_timeout_s = Path(log_path), startup_timeout_s
        self.process: subprocess.Popen | None = None
        self.startup_s: float | None = None

    def start(self) -> VLLMServer:
        """Launch vLLM and block until /health answers. Raises, with the log's error lines, if it can't."""
        if self.process is not None:
            raise RuntimeError("server already started")
        start = time.perf_counter()
        self._log = self.log_path.open("w")
        self.process = subprocess.Popen(
            self.command, stdout=self._log, stderr=subprocess.STDOUT, env={**os.environ, **self.env}
        )
        while time.perf_counter() - start < self.startup_timeout_s:
            if self.process.poll() is not None:
                self._log.close()
                raise RuntimeError(f"vLLM exited during startup:\n{self.log_tail()}")
            if self._healthy():
                self.startup_s = time.perf_counter() - start
                return self
            time.sleep(1)
        self.stop()
        raise TimeoutError(f"vLLM not ready after {self.startup_timeout_s} s:\n{self.log_tail()}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.process is not None and not self._log.closed:
            self._log.close()

    def __enter__(self) -> VLLMServer:
        return self if self.process is not None else self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def log_tail(self, lines: int = 60) -> str:
        """The log tail, preceded by earlier lines that mention an error (root causes come early)."""
        try:
            log = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return "(no log)"
        errors = [line for line in log[:-lines] if "Error" in line or "error:" in line][:40]
        return "\n".join([*errors, "...", *log[-lines:]] if errors else log[-lines:])
