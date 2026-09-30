"""Serving benchmarks: run a real inference server (vLLM) and measure it under load.

server.py     start/stop `vllm serve` inside the container
workloads.py  what requests look like (lengths, shared prefixes) and when they arrive
client.py     an async streaming client that timestamps every token
metrics.py    TTFT, TPOT, ITL, throughput, goodput, percentiles (stdlib only)
"""
