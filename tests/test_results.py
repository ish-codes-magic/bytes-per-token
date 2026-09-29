from fastserve.results import (
    SCHEMA_VERSION,
    append_jsonl,
    git_info,
    latest_run,
    make_record,
    read_jsonl,
    to_plain,
)


def test_make_record_carries_metadata():
    record = make_record(
        "demo", {"x": 1}, run_id="abc", config={"k": 2}, git={"commit": "c"}, env={"gpu": "g"}
    )
    assert record["schema"] == SCHEMA_VERSION
    assert record["experiment"] == "demo"
    assert record["run_id"] == "abc"
    assert record["metrics"] == {"x": 1}
    assert record["config"] == {"k": 2}
    assert record["git"] == {"commit": "c"}
    assert record["env"] == {"gpu": "g"}
    assert record["timestamp"].endswith("+00:00")  # always UTC


def test_make_record_collects_environment_by_default():
    env = make_record("demo", {}, run_id="abc")["env"]
    assert {"python", "os", "cpu", "cpu_count"} <= env.keys()


def test_environment_info_uses_only_plain_types():
    # Records are unpickled on a laptop without PyTorch: a str subclass like TorchVersion would break that.
    env = make_record("demo", {}, run_id="abc")["env"]
    assert all(type(value) in (str, int, float, bool, type(None)) for value in env.values()), env


def test_to_plain_strips_library_types():
    class Version(str):  # stands in for torch's TorchVersion
        pass

    plain = to_plain([{"torch": Version("2.14.0")}])
    assert type(plain[0]["torch"]) is str


def test_jsonl_round_trip_appends(tmp_path):
    path = tmp_path / "raw" / "out.jsonl"
    first = [make_record("a", {"i": i}, run_id="r1", env={}) for i in range(3)]
    assert append_jsonl(path, first) == 3
    append_jsonl(path, [make_record("b", {}, run_id="r2", env={})])
    records = read_jsonl(path)
    assert [r["experiment"] for r in records] == ["a", "a", "a", "b"]
    assert "\r" not in path.read_text(encoding="utf-8")  # LF line endings on every OS


def test_latest_run_selects_newest_run_id():
    old = [dict(make_record("a", {}, run_id="old", env={}), timestamp="2026-01-01T00:00:00+00:00")]
    new = [dict(make_record("a", {}, run_id="new", env={}), timestamp="2026-02-01T00:00:00+00:00")] * 2
    assert {r["run_id"] for r in latest_run(old + new)} == {"new"}
    assert latest_run([]) == []


def test_git_info_outside_a_repository(tmp_path):
    assert git_info(tmp_path) == {"commit": None, "dirty": None}
