import pytest
from trainsafe.checks.length import LengthCheck


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def test_baseline_set_on_first_run():
    check = LengthCheck()
    result = check.run([_words(80)] * 5)
    assert result["score"] == 1.0
    assert result["status"] == "ok"
    assert check.baseline_mean == 80.0


def test_normal_length_passes():
    check = LengthCheck()
    check.run([_words(80)] * 5)
    result = check.run([_words(90)] * 5)
    assert result["score"] == 1.0
    assert result["status"] == "ok"


def test_collapse_detected():
    check = LengthCheck()
    check.run([_words(80)] * 5)
    result = check.run([_words(3)] * 5)  # 3/80 = 0.037 < 0.5
    assert result["score"] == 0.0
    assert result["status"] == "warn"
    assert "collapsed" in result["message"].lower()


def test_spike_detected():
    check = LengthCheck()
    check.run([_words(80)] * 5)
    result = check.run([_words(400)] * 5)  # 400/80 = 5.0 > 3.0
    assert result["score"] == 0.0
    assert result["status"] == "warn"
    assert "spike" in result["message"].lower()


def test_empty_outputs_skip():
    check = LengthCheck()
    result = check.run([])
    assert result["score"] == 1.0
    assert result["status"] == "skip"


def test_reset_clears_baseline():
    check = LengthCheck()
    check.run([_words(80)] * 5)
    check.reset()
    assert check.baseline_mean is None


def test_length_ema_adapts_baseline():
    check = LengthCheck(ema_alpha=0.5)
    # First run sets baseline to 10
    check.run(["word " * 10] * 5)  # 10 words each (trailing space doesn't add a word after split)
    original_baseline = check.baseline_mean
    # Second passing run at 20 words — baseline should shift toward 20
    check.run(["word " * 20] * 5)
    assert check.baseline_mean > original_baseline
    assert check.baseline_mean < 20  # EMA, not full update


def test_length_frozen_when_ema_zero():
    check = LengthCheck(ema_alpha=0.0)
    check.run(["word " * 10] * 5)
    baseline = check.baseline_mean
    check.run(["word " * 20] * 5)
    assert check.baseline_mean == baseline  # frozen
