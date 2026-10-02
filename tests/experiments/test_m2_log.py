"""What run_serving reads from vLLM's startup log."""

import pytest

pytest.importorskip("torch")

from fastserve.experiments.m2 import _from_log, cuda_graphs  # noqa: E402

LOG = """\
INFO 10-01 12:00:01 [gpu_model_runner.py:1234] Model loading took 0.7512 GiB and 3.20 seconds
INFO 10-01 12:00:02 [compressed_tensors_wNa16.py:88] Using MarlinLinearKernel for CompressedTensorsWNA16
INFO 10-01 12:00:02 [compressed_tensors_wNa16.py:88] Using MarlinLinearKernel for CompressedTensorsWNA16
INFO 10-01 12:00:09 [gpu_worker.py:300] Available KV cache memory: 19.25 GiB
INFO 10-01 12:00:09 [kv_cache_utils.py:900] GPU KV cache size: 180,224 tokens
INFO 10-01 12:00:09 [kv_cache_utils.py:901] Maximum concurrency for 40,960 tokens per request: 4.40x
INFO 10-01 12:00:03 [cuda.py:400] Using FLASHINFER attention backend out of potential backends: [...]
INFO 10-01 12:00:03 [cache.py:70] Using fp8 data type to store kv cache. It reduces the GPU memory footprint
"""


class FakeServer:
    def __init__(self, path):
        self.log_path = path


def test_startup_facts_come_from_the_log(tmp_path):
    path = tmp_path / "vllm.log"
    path.write_text(LOG, encoding="utf-8")
    facts = _from_log(FakeServer(path))
    assert facts["kv_cache_tokens"] == 180224
    assert facts["model_memory_gib"] == pytest.approx(0.7512)
    assert facts["kv_cache_memory_gib"] == pytest.approx(19.25)
    assert facts["kernels"] == ["Using MarlinLinearKernel for CompressedTensorsWNA16"]
    assert facts["max_concurrency"] == pytest.approx(4.40)
    assert facts["attention_backend"] == "FLASHINFER" and facts["kv_cache_dtype"] == "fp8"


def test_missing_lines_are_none(tmp_path):
    path = tmp_path / "vllm.log"
    path.write_text("nothing useful\n", encoding="utf-8")
    facts = _from_log(FakeServer(path))
    assert facts["kv_cache_tokens"] is None and facts["kernels"] == []


BARS = (
    "Capturing CUDA graphs (PIECEWISE):  98%|#########8| 50/51 [00:06<00:00,  8.07it/s]\r"
    "Capturing CUDA graphs (PIECEWISE): 100%|##########| 51/51 [00:07<00:00,  7.18it/s]\n"
    "Capturing CUDA graphs (FULL): 100%|##########| 2/2 [00:00<00:00, 11.97it/s]\n"  # the memory profiler
    "Capturing CUDA graphs (FULL):  94%|#########4| 33/35 [00:01<00:00, 20.20it/s]\r"
    "Capturing CUDA graphs (FULL): 100%|##########| 35/35 [00:01<00:00, 19.63it/s]\n"
    "Capturing decode CUDA graphs (FULL): 100%|##########| 35/35 [00:00<00:00, 50.17it/s]\n"
)
WARNING = (
    "WARNING 10-02 07:41:52 [compilation.py:1473] CUDAGraphMode.FULL_AND_PIECEWISE is not supported with "
    "spec-decode for attention backend FlashInferBackend (support: "
    "AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE\n"
)


def test_captured_graphs_are_counted_per_kind():
    graphs = cuda_graphs(BARS)
    assert graphs == {"captured": {"piecewise": 51, "full": 35, "decode full": 35}, "fallback": None}


def test_a_fallback_keeps_vllms_own_words():
    graphs = cuda_graphs(WARNING + BARS.split("\n")[0] + "\n")  # only the piecewise bar: no full graph
    assert graphs["captured"] == {"piecewise": 51}
    assert graphs["fallback"].startswith("CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode")
    assert graphs["fallback"].endswith("setting cudagraph_mode=PIECEWISE")
    assert cuda_graphs("nothing useful") == {"captured": {}, "fallback": None}
