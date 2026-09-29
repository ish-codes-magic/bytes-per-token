"""What does it cost just to *launch* a kernel? This is the target of lever 3.

We launch many tiny kernels (each adds 1 to a single number) and divide the total time by the count. Each
kernel does almost no work, so the time is pure overhead: Python, PyTorch dispatch and the CUDA launch.
Then we capture the same kernels in a CUDA graph and replay them with a single call. The difference is the
overhead CUDA graphs remove, which is why vLLM uses them for decode.
"""

from __future__ import annotations

from typing import Any

from fastserve.timing import cuda_time_ms, summarize


def launch_overhead(n_ops: int = 1000, *, iters: int = 20) -> dict[str, Any]:
    import torch

    x = torch.zeros(1, device="cuda")

    def many_tiny_kernels() -> None:
        for _ in range(n_ops):
            x.add_(1.0)

    eager = summarize(cuda_time_ms(many_tiny_kernels, warmup=3, iters=iters))

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # PyTorch requires a warm-up run on a side stream before capture
        many_tiny_kernels()
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        many_tiny_kernels()
    replay = summarize(cuda_time_ms(graph.replay, warmup=3, iters=iters))

    return {
        "n_ops": n_ops,
        "eager_us_per_kernel": eager.median_ms * 1e3 / n_ops,
        "graph_us_per_kernel": replay.median_ms * 1e3 / n_ops,
        "eager_timing": eager.to_dict(),
        "graph_timing": replay.to_dict(),
    }
