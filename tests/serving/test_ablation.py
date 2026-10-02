"""The ablation plan: labels are sets of techniques, and the expansion is what vLLM is started with."""

import pytest

from fastserve.serving.ablation import differs_by_one, expand, factorial, label_of, labels, letters, server

CONFIG = {
    "techniques": {
        "w": {"name": "FP8 weights", "path": "me/{model}-FP8"},
        "a": {"name": "INT4 weights", "path": "me/{model}-INT4"},
        "k": {"name": "FP8 KV", "args": ["--kv-cache-dtype", "fp8"]},
        "p": {"name": "prefix caching", "args": ["--cache"], "off_args": ["--no-cache"]},
        "s": {"name": "speculation", "speculative": {"Qwen/Qwen3-1.7B": {"method": "eagle3"}}},
    },
    "common_args": ["--details"],
    "plan": {
        "Qwen/Qwen3-1.7B": {
            "factorial": ["w", "k"],
            "combos": ["wk", "akps"],
            "repeats": ["base"],
            "ladder": ["base", "wk"],
        }
    },
    "loads": {"chat": [], "long": []},
    "only_for": {"long": "ladder"},
}


def test_labels_are_sets_of_letters():
    assert letters("base") == "" and letters("wk") == "wk" and letters("wkps-r2") == "wkps"
    assert label_of({"k", "w"}, "wakps") == "wk" and label_of(set(), "wakps") == "base"
    assert factorial(["w", "k", "p"]) == ["base", "w", "k", "p", "wk", "wp", "kp", "wkp"]
    assert len(factorial(["w", "k", "p", "s"])) == 16  # the ladder, leave-one-out and every pair
    assert differs_by_one("wk", "wkp") == "p" and differs_by_one("base", "w") == "w"
    assert differs_by_one("w", "k") is None and differs_by_one("wk", "wk-r2") is None


def test_plan_expands_to_servers_without_duplicates():
    plan = CONFIG["plan"]["Qwen/Qwen3-1.7B"]
    assert labels(plan) == ["base", "w", "k", "wk", "akps", "base-r2"]  # "wk" listed twice runs once
    servers = {s["label"]: s for s in expand(CONFIG)}
    assert servers["base"] == {
        "name": "1.7b-base",
        "label": "base",
        "model": "Qwen/Qwen3-1.7B",
        "args": ["--details", "--no-cache"],  # a technique that is off may need saying so
        "workloads": ["chat", "long"],  # in the ladder: it also runs the ladder-only workload
    }
    full = servers["akps"]
    assert full["path"] == "me/Qwen3-1.7B-INT4" and full["speculative"] == {"method": "eagle3"}
    assert full["args"] == ["--details", "--kv-cache-dtype", "fp8", "--cache"]
    assert full["workloads"] == ["chat"]
    assert (
        servers["base-r2"]["args"] == servers["base"]["args"] and servers["base-r2"]["name"] == "1.7b-base-r2"
    )
    assert servers["w"]["workloads"] == ["chat"] and servers["wk"]["workloads"] == ["chat", "long"]


def test_bad_labels_are_refused():
    plan = CONFIG["plan"]["Qwen/Qwen3-1.7B"]
    with pytest.raises(ValueError, match="two checkpoints"):
        server("Qwen/Qwen3-1.7B", "wa", plan, CONFIG)
    with pytest.raises(ValueError, match="undefined techniques"):
        server("Qwen/Qwen3-1.7B", "wz", plan, CONFIG)


def test_the_real_plan_is_one_change_at_a_time():
    """The committed config: every ladder step adds one technique, and the 1.7B grid is complete."""
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    path = Path(__file__).parents[2] / "benchmarks" / "configs" / "m8_ablation.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    servers = expand(config)
    assert len({s["name"] for s in servers}) == len(servers) == 34  # 30 planned + 4 controls
    large = [s["label"] for s in servers if s["model"] == "Qwen/Qwen3-1.7B"]
    assert set(factorial(["w", "k", "p", "s"])) <= set(large)
    ladder = ["base", "w", "wk", "wkp", "wkps"]
    assert [differs_by_one(a, b) for a, b in zip(ladder, ladder[1:], strict=False)] == ["w", "k", "p", "s"]
    assert differs_by_one("w", "wf") == "f" and differs_by_one("wkps", "akps") is None  # a swap, not a step
    for s in servers:
        assert ("long_32k" in s["workloads"]) == (s["label"] in config["plan"][s["model"]]["ladder"])
