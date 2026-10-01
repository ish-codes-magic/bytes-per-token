import json

import pytest
import torch

from fastserve.quant.compressed import dense_state_dict, dequantize_linear, load_dense, pack_int, unpack_int


@pytest.mark.parametrize("num_bits", [2, 4, 8])
def test_pack_unpack_round_trip_covers_every_value(num_bits):
    g = torch.Generator().manual_seed(0)
    low, high = -(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1
    q = torch.randint(low, high + 1, (5, 64), generator=g, dtype=torch.int8)
    q[0, :2] = torch.tensor([low, high])  # the extremes set the sign bit of the word
    assert torch.equal(unpack_int(pack_int(q, num_bits), 64, num_bits), q)
    # a row that doesn't fill its last word is padded, then cut back
    assert torch.equal(unpack_int(pack_int(q[:, :37], num_bits), 37, num_bits), q[:, :37])


def test_pack_layout_matches_compressed_tensors():
    # value i at bits 4i..4i+3, stored as q + 8: [-8, -7, ..., -1] → nibbles 0..7 → 0x76543210
    q = torch.arange(-8, 0, dtype=torch.int8).reshape(1, 8)
    assert pack_int(q, 4).item() == 0x76543210
    # [0..7] → nibbles 8..15 → 0xFEDCBA98, which as a signed int32 is negative
    q = torch.arange(0, 8, dtype=torch.int8).reshape(1, 8)
    assert pack_int(q, 4).item() == 0xFEDCBA98 - 2**32


def test_dequantize_uses_each_rows_group_scale():
    q = torch.tensor([[1, -2, 3, -4], [7, 0, -8, 5]], dtype=torch.int8)
    scale = torch.tensor([[0.5, 2.0], [1.0, 0.25]], dtype=torch.bfloat16)  # groups of 2 columns
    w = dequantize_linear(
        {"weight_packed": pack_int(q, 4), "weight_scale": scale, "weight_shape": torch.tensor([2, 4])}
    )
    expected = torch.tensor([[0.5, -1.0, 6.0, -8.0], [7.0, 0.0, -2.0, 1.25]], dtype=torch.bfloat16)
    assert torch.equal(w, expected)


def test_dequantize_fp8_and_int8_per_channel():
    x = torch.tensor([[1.5, -0.25], [448.0, 2.0]])
    scale = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    fp8 = dequantize_linear({"weight": x.to(torch.float8_e4m3fn), "weight_scale": scale})
    assert torch.equal(fp8, torch.tensor([[3.0, -0.5], [224.0, 1.0]], dtype=torch.bfloat16))
    int8 = dequantize_linear(
        {"weight": torch.tensor([[3, -127]], dtype=torch.int8), "weight_scale": scale[:1]}
    )
    assert torch.equal(int8, torch.tensor([[6.0, -254.0]], dtype=torch.bfloat16))


def test_state_dict_and_folder_round_trip(tmp_path):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(1)
    q = torch.randint(-8, 8, (4, 256), generator=g, dtype=torch.int8)
    scale = torch.rand(4, 2, generator=g).to(torch.bfloat16)
    state = {
        "model.embed_tokens.weight": torch.ones(3, 4, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight_packed": pack_int(q, 4),
        "model.layers.0.mlp.down_proj.weight_scale": scale,
        "model.layers.0.mlp.down_proj.weight_shape": torch.tensor([4, 256]),
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    config = {"quantization_config": {"config_groups": {"group_0": {"weights": {"num_bits": 4}}}}}
    dense = dense_state_dict(state, config["quantization_config"])
    assert set(dense) == {
        "model.embed_tokens.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
    }
    expected = (q.float() * scale.float().repeat_interleave(128, dim=1)).to(torch.bfloat16)
    assert torch.equal(dense["model.layers.0.mlp.down_proj.weight"], expected)

    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    loaded = load_dense(tmp_path)
    assert all(torch.equal(loaded[k], dense[k]) for k in dense)
