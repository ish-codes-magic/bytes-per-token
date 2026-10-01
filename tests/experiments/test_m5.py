"""M5's reference tasks, end to end on a tiny random model (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.experiments.m3 import Bench  # noqa: E402
from fastserve.experiments.m5 import kv_kl, kv_needle, kv_spec, kv_stats  # noqa: E402

INT4 = {"name": "int4", "keys": {"bits": 4, "group": 16}, "values": {"bits": 4, "group": 16}}


class CharTokenizer:
    """Just enough of a Hugging Face tokenizer for the needle grid: one token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 250 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ):
        return messages[0]["content"]

    def convert_tokens_to_ids(self, tokens):
        return [250 + i for i, _ in enumerate(tokens)]


@pytest.fixture
def bench(tiny_model):
    windows = torch.randint(0, 250, (2, 32), generator=torch.Generator().manual_seed(1))
    return Bench("tiny", tiny_model, CharTokenizer(), windows, {})


def test_specs_parse_from_config():
    spec = kv_spec(
        {"name": "kivi", "keys": {"bits": 4, "axis": "channel", "group": 32}, "values": {"bits": 4}}
    )
    assert spec.keys.axis == "channel" and spec.values.group == 128 and spec.window is None
    assert kv_spec({"name": "streaming", "sinks": 4, "window": 1020}).kept_tokens(32768) == 1024


def test_kl_is_zero_for_a_bf16_cache_and_positive_when_quantized(bench):
    assert kv_kl(bench, {"name": "bf16"})["mean_kl"] == 0.0
    int4 = kv_kl(bench, INT4)
    assert int4["mean_kl"] > 0 and int4["bits_per_element"] == 4 + 32 / 16
    assert all(layer.self_attn.kv_policy is None for layer in bench.ref.model.layers)  # left clean


def test_needle_grid_runs_through_the_chunked_cache(bench):
    cfg = {"lengths": [40], "depths": [0.0, 1.0], "secrets": ["42"], "chunk": 16, "max_new_tokens": 3}
    out = kv_needle(bench, INT4, cfg)
    assert len(out["cells"]) == 2 and 0.0 <= out["pass_rate"] <= 1.0
    assert all(set(c) == {"length", "depth", "secret", "passed", "answer"} for c in out["cells"])


def test_stats_profile_every_layer(bench):
    out = kv_stats(bench, windows=2, layers=[0])
    assert len(out["key_ratio"]) == len(bench.ref.model.layers) and all(r >= 1 for r in out["key_ratio"])
    assert len(out["profiles"]["0"]["keys"]) == bench.config.num_kv_heads
    assert len(out["profiles"]["0"]["keys"][0]) == bench.config.head_dim
