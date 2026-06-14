import pytest
from trainsafe.checks.language import LanguageCheck


def test_baseline_is_set_on_first_run():
    check = LanguageCheck()
    result = check.run(["Hello world, this is a test sentence in English."] * 5)
    assert result["status"] == "ok"
    assert result["score"] == 1.0
    assert check.baseline_language == "en"


def test_consistent_language_passes():
    check = LanguageCheck()
    check.run(["Hello world, this is a test sentence in English."] * 5)
    result = check.run(["Another English sentence here, everything looks fine today."] * 5)
    assert result["score"] == 1.0
    assert result["status"] == "ok"


def test_language_drift_fails():
    check = LanguageCheck()
    english = ["Hello world, this is a long English sentence that should be detectable."] * 5
    check.run(english)
    chinese = ["这是一个很长的中文句子，用来测试语言检测功能是否正常工作。"] * 5
    result = check.run(chinese)
    assert result["score"] == 0.0
    assert result["status"] == "fail"
    assert "drift" in result["message"].lower()


def test_empty_outputs_are_skipped():
    check = LanguageCheck()
    result = check.run([])
    assert result["score"] == 1.0
    assert result["status"] == "skip"


def test_short_outputs_are_skipped():
    check = LanguageCheck()
    result = check.run(["Hi"] * 5)
    assert result["score"] == 1.0
    assert result["status"] == "skip"


def test_reset_clears_baseline():
    check = LanguageCheck()
    check.run(["Hello world, this is a test sentence in English."] * 5)
    assert check.baseline_language == "en"
    check.reset()
    assert check.baseline_language is None
    assert check._pending_language is None
    assert check._pending_count == 0


_EN = ["Hello world, this is a long English sentence that should be detectable."] * 5
_ZH = ["这是一个很长的中文句子，用来测试语言检测功能是否正常工作。"] * 5


def test_language_grace_period_updates_baseline():
    check = LanguageCheck(baseline_grace=3)
    check.run(_EN)
    assert check.baseline_language == "en"

    r1 = check.run(_ZH)
    assert r1["status"] == "fail"
    r2 = check.run(_ZH)
    assert r2["status"] == "fail"
    r3 = check.run(_ZH)
    assert r3["status"] == "ok"
    assert "updated" in r3["message"]
    assert check.baseline_language == "zh-cn"


def test_language_grace_resets_on_different_language():
    check = LanguageCheck(baseline_grace=3)
    check.run(_EN)

    check.run(_ZH)   # pending zh-cn count = 1
    check.run(_EN)   # back to en, resets pending
    check.run(_ZH)   # pending zh-cn count = 1 again
    r = check.run(_ZH)  # count = 2, still warns
    assert r["status"] == "fail"
    assert check.baseline_language == "en"
