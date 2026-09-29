from fastserve.report.tables import hw_summary, markdown_table


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
