import pytest
from trainsafe.checks.format import FormatCheck, detect_format


def test_detect_json():
    assert detect_format('{"key": "value", "number": 42}') == "json"


def test_detect_json_array():
    assert detect_format('["a", "b", "c"]') == "json"


def test_detect_markdown():
    assert detect_format("# Title\n\nSome **bold** text here") == "markdown"


def test_detect_plain():
    assert detect_format("Just a plain text sentence with nothing special.") == "plain"


def test_empty_is_plain():
    assert detect_format("") == "plain"
    assert detect_format("   ") == "plain"


def test_format_consistent():
    check = FormatCheck()
    outputs = ['{"name": "Alice", "age": 30}', '{"name": "Bob", "age": 25}']
    check.run(outputs)  # baseline
    result = check.run(['{"name": "Charlie", "age": 22}', '{"x": 1}'])
    assert result["score"] == 1.0
    assert result["status"] == "ok"


def test_format_drift_detected():
    check = FormatCheck()
    check.run(['{"key": "value"}', '{"a": 1}'])  # baseline: json
    result = check.run(["Just plain text output.", "Another plain sentence here."])
    assert result["score"] == 0.0
    assert result["status"] == "warn"
    assert "drift" in result["message"].lower()


def test_reset_clears_baseline():
    check = FormatCheck()
    check.run(['{"key": "value"}'])
    assert check.baseline_format == "json"
    check.reset()
    assert check.baseline_format is None


def test_empty_outputs_skip():
    check = FormatCheck()
    result = check.run([])
    assert result["status"] == "skip"


def test_format_grace_period_updates_baseline():
    check = FormatCheck(baseline_grace=3)
    check.run(["just plain text"] * 3)  # baseline = plain
    assert check.baseline_format == "plain"

    # 2 consecutive json checkpoints — still warns
    r1 = check.run(['{"key": "value"}'] * 3)
    assert r1["status"] == "warn"
    r2 = check.run(['{"key": "value"}'] * 3)
    assert r2["status"] == "warn"

    # 3rd consecutive json — baseline updates
    r3 = check.run(['{"key": "value"}'] * 3)
    assert r3["status"] == "ok"
    assert check.baseline_format == "json"


def test_format_grace_resets_on_different_format():
    check = FormatCheck(baseline_grace=3)
    check.run(["plain text"] * 3)   # baseline = plain
    check.run(['{"a": 1}'] * 3)    # pending json count = 1
    check.run(["# markdown header"] * 3)  # switches to markdown, resets count
    check.run(['{"a": 1}'] * 3)    # back to json, count = 1 again
    r = check.run(['{"a": 1}'] * 3)  # count = 2, still warns
    assert r["status"] == "warn"
    assert check.baseline_format == "plain"
