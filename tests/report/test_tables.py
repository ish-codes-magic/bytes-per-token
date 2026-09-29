from fastserve.report.tables import hw_summary, markdown_table, prediction_table


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
