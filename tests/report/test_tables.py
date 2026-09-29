from fastserve.report.tables import decode_matmul_table, hw_summary, markdown_table, prediction_table


def test_markdown_table_shape():
    assert markdown_table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_hw_summary_puts_measured_next_to_datasheet(probe_records):
    text = hw_summary(probe_records)
    assert text.startswith("*NVIDIA L4")
    assert "commit `0123456`" in text
    assert "| Memory bandwidth, read (GB/s) | 280 | 300 | 93% |" in text
    assert "| BF16 matmul peak (TFLOP/s) | 90.0 | 121 | 74% |" in text
    assert "| Kernel launch, eager (µs per kernel) | 6.00 | — | — |" in text
    assert "| Power, streaming memory (W) | — | — | — |" in text  # missing readings stay visible as dashes
    # A power-capped GPU runs below its maximum clock; the table shows it and scales the datasheet peak to it.
    assert "| SM clock, sustained BF16 matmul (MHz) | 1,020 | 2,040 | 50% |" in text
    assert "| BF16 datasheet peak at that clock (TFLOP/s) | 60.5 | 121 | 50% |" in text
    assert "58 → 69" in text


def test_prediction_table_verdicts():
    predictions = {
        "written_in_commit": "abc1234",
        "predictions": [
            {"id": "a", "label": "A", "low": 1, "high": 2},
            {"id": "b", "label": "B", "low": 1, "high": 2},
            {"id": "c", "label": "C", "low": 1, "high": 2},
            {"id": "d", "label": "D", "point": 10},
            {"id": "e", "label": "E", "point": 10},
        ],
    }
    text = prediction_table(predictions, {"a": 1.5, "b": 0.5, "c": 3.0, "d": 15.0, "e": None})
    assert "commit `abc1234`" in text
    assert "| A | 1 – 2 | 1.5 | within range |" in text
    assert "| B | 1 – 2 | 0.5 | below range |" in text
    assert "| C | 1 – 2 | 3 | above range |" in text
    assert "| D | ~10 | 15 | 1.5× the prediction |" in text
    assert "| E | ~10 | — | — |" in text


def test_decode_matmul_table_shows_how_fast_weights_stream(probe_records):
    # Fake M=1, N=K=1024 BF16 row: a 2 MiB weight in ~16.9 µs is ~124 GB/s, 44% of the fake 280 GB/s.
    text = decode_matmul_table(probe_records)
    assert "| 1024 × 1024 | 2 | 17 | 124 | 44% |" in text
    assert text.count("\n") == 3  # header + separator + the two BF16 M=1 shapes


def test_model_facts_for_qwen3(probe_records):
    from pathlib import Path

    from fastserve.engine.config import ModelConfig
    from fastserve.report.tables import model_facts

    path = Path(__file__).parents[2] / "benchmarks" / "models" / "Qwen3-0.6B.config.json"
    text = model_facts(ModelConfig.from_pretrained_json(path), "Qwen3-0.6B", probe_records)
    assert "| Parameters | 596 M |" in text
    assert "| KV cache per token (BF16) | 112 KiB |" in text
    assert (
        "| Batch-1 ceiling = measured read bandwidth ÷ bytes per step | 235 tokens/s |" in text
    )  # 280 / 1.192
