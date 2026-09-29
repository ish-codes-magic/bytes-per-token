"""Markdown tables built from result records. Docs include them through generated blocks (render.py)."""

from __future__ import annotations

from typing import Any

from fastserve.hw.analysis import gpu_spec, measured_bandwidth, measured_peak_flops, metrics_of, single
from fastserve.perfmodel.roofline import ridge_point

Records = list[dict[str, Any]]
DASH = "—"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def provenance(records: Records) -> str:
    """One italic line saying where the numbers came from."""
    env, git = records[0]["env"], records[0]["git"]
    commit = (git.get("commit") or "unknown")[:7] + (" (uncommitted changes)" if git.get("dirty") else "")
    return (
        f"*{env.get('gpu', 'unknown GPU')} · driver {env.get('driver')} · CUDA {env.get('cuda_runtime')} · "
        f"PyTorch {env.get('torch')} · Triton {env.get('triton')} · host CPU {env.get('cpu')} · "
        f"run `{records[0]['run_id']}` · commit `{commit}` · {records[0]['timestamp']}*"
    )


def decode_matmul_table(records: Records) -> str:
    """BF16 matmuls at M=1 (decode at batch 1): how fast each weight really streams from memory."""
    read_bw = measured_bandwidth(records, method="read")
    decode_shaped = sorted(
        (m for m in metrics_of(records, "matmul") if m["format"] == "bf16" and m["m"] == 1 and "timing" in m),
        key=lambda m: m["n"],
    )
    rows = []
    for m in decode_shaped:
        weight_bytes = 2 * m["n"] * m["k"]  # BF16: 2 bytes per weight, every weight read once
        seconds = m["timing"]["median_ms"] / 1e3
        streamed = weight_bytes / seconds
        rows.append(
            [
                f"{m['n']} × {m['k']}",
                _num(weight_bytes, 2**20),
                _num(seconds, 1e-6),
                _num(streamed, 1e9),
                _ratio(streamed, read_bw),
            ]
        )
    headers = [
        "Weight (N × K)",
        "Size (MiB)",
        "Time (µs)",
        "Streamed at (GB/s)",
        "vs measured read bandwidth",
    ]
    return markdown_table(headers, rows)


def _sig(x: float) -> str:
    """Three significant figures: enough to compare against a back-of-envelope prediction."""
    return f"{x:,.3g}" if abs(x) < 1000 else f"{x:,.0f}"


def prediction_table(predictions: dict[str, Any], observed: dict[str, float | None]) -> str:
    """Predicted vs measured, one row per prediction, with an automatic verdict.

    A prediction is either a range ("low"/"high") or a point estimate ("point").
    Ranges get within/below/above; point estimates get the measured/predicted ratio.
    """
    rows = []
    for p in predictions["predictions"]:
        value = observed.get(p["id"])
        if "point" in p:
            predicted = f"~{_sig(p['point'])}"
            verdict = (
                f"{value / p['point']:.2g}× the prediction" if value is not None and p["point"] else DASH
            )
        else:
            predicted = f"{_sig(p['low'])} – {_sig(p['high'])}"
            if value is None:
                verdict = DASH
            elif value < p["low"]:
                verdict = "below range"
            elif value > p["high"]:
                verdict = "above range"
            else:
                verdict = "within range"
        rows.append([p["label"], predicted, DASH if value is None else _sig(value), verdict])
    note = (
        f"*Predictions written in commit `{predictions['written_in_commit']}`, before the first measurement.*"
    )
    return f"{note}\n\n" + markdown_table(["Quantity", "Predicted", "Measured", "Verdict"], rows)


def _num(x: float | None, scale: float = 1.0, digits: int = 0) -> str:
    return DASH if x is None else f"{x / scale:,.{digits}f}"


def _ratio(measured: float | None, promised: float | None) -> str:
    return f"{measured / promised:.0%}" if measured and promised else DASH


def hw_summary(records: Records) -> str:
    """M0 headline table: every measured ceiling next to its datasheet value."""
    spec = gpu_spec(records) or {}
    spec_bw = spec.get("bandwidth")
    spec_peaks = spec.get("peak_flops", {})
    peaks = measured_peak_flops(records)
    read_bw = measured_bandwidth(records, method="read")
    copy_bw = measured_bandwidth(records, method="copy")

    rows = [
        ["Memory bandwidth, read (GB/s)", _num(read_bw, 1e9), _num(spec_bw, 1e9), _ratio(read_bw, spec_bw)],
        ["Memory bandwidth, copy (GB/s)", _num(copy_bw, 1e9), _num(spec_bw, 1e9), _ratio(copy_bw, spec_bw)],
    ]
    for fmt in ("bf16", "fp16", "fp8", "int8"):
        unit = "TOPS" if fmt == "int8" else "TFLOP/s"
        measured, promised = peaks.get(fmt), spec_peaks.get(fmt)
        rows.append(
            [
                f"{fmt.upper()} matmul peak ({unit})",
                _num(measured, 1e12, 1),
                _num(promised, 1e12),
                _ratio(measured, promised),
            ]
        )
    for fmt in ("bf16", "fp8"):
        measured = ridge_point(peak_flops=peaks[fmt], bandwidth=read_bw) if fmt in peaks and read_bw else None
        promised = (
            ridge_point(peak_flops=spec_peaks[fmt], bandwidth=spec_bw)
            if fmt in spec_peaks and spec_bw
            else None
        )
        rows.append([f"{fmt.upper()} ridge point (FLOPs/byte)", _num(measured), _num(promised), DASH])

    launch = single(records, "launch_overhead")
    if launch:
        rows.append(
            [
                "Kernel launch, eager (µs per kernel)",
                _num(launch["eager_us_per_kernel"], digits=2),
                DASH,
                DASH,
            ]
        )
        rows.append(
            [
                "Kernel launch, CUDA graph (µs per kernel)",
                _num(launch["graph_us_per_kernel"], digits=2),
                DASH,
                DASH,
            ]
        )

    power = single(records, "power") or {}
    for workload, label in (
        ("idle", "idle"),
        ("copy_1gib", "streaming memory"),
        ("matmul_bf16_8192", "BF16 matmul"),
    ):
        reading = power.get(workload)
        promised = spec.get("power_w") if workload == "matmul_bf16_8192" else None
        rows.append(
            [
                f"Power, {label} (W)",
                _num(reading["mean_w"] if reading else None),
                _num(promised),
                _ratio(reading["mean_w"] if reading else None, promised),
            ]
        )

    # A power-capped GPU lowers its clock under sustained load; datasheet peaks assume the maximum clock.
    start = (single(records, "gpu_info") or {}).get("telemetry") or {}
    end = (single(records, "gpu_info_end") or {}).get("telemetry") or {}
    max_clock = start.get("max_sm_clock_mhz")
    loaded = power.get("matmul_bf16_8192") or {}
    clock = loaded.get("mean_sm_clock_mhz")
    rows.append(
        ["SM clock, sustained BF16 matmul (MHz)", _num(clock), _num(max_clock), _ratio(clock, max_clock)]
    )
    scaled = spec_peaks["bf16"] * clock / max_clock if "bf16" in spec_peaks and clock and max_clock else None
    rows.append(
        [
            "BF16 datasheet peak at that clock (TFLOP/s)",
            _num(scaled, 1e12, 1),
            _num(spec_peaks.get("bf16"), 1e12),
            _ratio(scaled, spec_peaks.get("bf16")),
        ]
    )
    rows.append(
        [
            "Temperature start → end (°C)",
            f"{start.get('temperature_c', DASH)} → {end.get('temperature_c', DASH)}",
            DASH,
            DASH,
        ]
    )

    table = markdown_table(["Quantity", "Measured", "Datasheet", "Measured / datasheet"], rows)
    return f"{provenance(records)}\n\n{table}"
