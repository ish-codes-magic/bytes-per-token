"""nanoserve vs Hugging Face on the real Qwen3-0.6B in BF16 (weights from the Modal Volume)."""

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def model_dir():
    from fastserve.engine.loader import model_dir as local_model_dir

    try:
        return local_model_dir(MODEL)
    except Exception as err:  # keep the real reason visible in the skip message
        pytest.skip(f"{MODEL} isn't available ({err!r}); run `modal run infra/modal_app.py::fetch`")


def test_bf16_logits_track_hugging_face(model_dir):
    from fastserve.engine.loader import load_pretrained
    from fastserve.experiments.m1 import hf_parity

    ours = load_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
    rows = hf_parity(model_dir, ours, ["The capital of France is", "def add(a, b):\n    return"])
    # A regression guard against gross bugs; the measured agreement is recorded by the M1 experiment.
    assert min(r["top1_agreement"] for r in rows) >= 0.95
    assert max(r["mean_kl_hf_to_ours"] for r in rows) < 0.01
