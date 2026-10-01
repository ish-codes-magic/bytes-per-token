from types import SimpleNamespace

import pytest

from fastserve.kv.sizing import (
    BF16,
    KVSpec,
    TensorQuant,
    bytes_per_sequence,
    bytes_per_token,
    max_sequences,
    max_tokens,
)

QWEN3 = SimpleNamespace(num_layers=28, num_kv_heads=8, head_dim=128)  # both Qwen3-0.6B and Qwen3-1.7B


def test_bf16_is_112_kib_per_token_for_qwen3():
    assert bytes_per_token(QWEN3) == 2 * 28 * 8 * 128 * 2 == 112 * 1024


def test_quantized_formats_pay_for_their_scales():
    fp8 = KVSpec("FP8", keys=TensorQuant(8, kind="fp8"), values=TensorQuant(8, kind="fp8"))
    assert bytes_per_token(QWEN3, fp8) == bytes_per_token(QWEN3) / 2
    # INT4 KIVI: keys per channel over 32 tokens, values per token over 128 elements, scale + zero-point each
    kivi = KVSpec(
        "INT4 KIVI",
        keys=TensorQuant(4, axis="channel", group=32),
        values=TensorQuant(4, axis="token", group=128),
    )
    assert kivi.keys.bits_per_element() == 4 + 32 / 32
    assert kivi.values.bits_per_element() == 4 + 32 / 128
    assert bytes_per_token(QWEN3, kivi) == pytest.approx(bytes_per_token(QWEN3) * (5 + 4.25) / 2 / 16)


def test_capacity_and_eviction():
    memory = 18 * 2**30
    assert max_tokens(QWEN3, memory) == memory // (112 * 1024)
    assert max_sequences(QWEN3, memory, 8192) == max_tokens(QWEN3, memory) // 8192
    streaming = KVSpec("StreamingLLM", sinks=4, window=1020)
    # eviction caps each sequence at sinks + window tokens, however long the context
    assert bytes_per_sequence(QWEN3, 32768, streaming) == 1024 * bytes_per_token(QWEN3)
    assert bytes_per_sequence(QWEN3, 100, streaming) == 100 * bytes_per_token(QWEN3)
    assert max_sequences(QWEN3, memory, 32768, streaming) == memory // (1024 * bytes_per_token(QWEN3))
    assert max_sequences(QWEN3, memory, 32768, BF16) == memory // (32768 * bytes_per_token(QWEN3))
