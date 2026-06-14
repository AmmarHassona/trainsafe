import pytest
from trainsafe.checks.repetition import RepetitionCheck, _has_repetition


def _repeated(phrase: str, n: int) -> str:
    return " ".join([phrase] * n)


def test_clean_output_passes():
    check = RepetitionCheck()
    outputs = [
        "The quick brown fox jumps over the lazy dog near the river bank today.",
        "Machine learning models can be fine-tuned on custom datasets for better performance.",
        "Natural language processing is a fascinating field of artificial intelligence research.",
    ]
    result = check.run(outputs)
    assert result["score"] == 1.0
    assert result["status"] == "ok"


def test_repetitive_output_flagged():
    check = RepetitionCheck()
    repetitive = _repeated("the model said the same thing over and over", 8)
    outputs = [repetitive] * 5
    result = check.run(outputs)
    assert result["score"] < 1.0
    assert result["status"] == "warn"
    assert "repetition" in result["message"].lower()


def test_ngram_detection_helper_clean():
    assert not _has_repetition("The quick brown fox jumps over the lazy dog in the field")


def test_ngram_detection_helper_repeated():
    assert _has_repetition(_repeated("the model keeps saying the same thing again", 6))


def test_short_output_not_flagged():
    assert not _has_repetition("hello world hi there")


def test_empty_outputs():
    check = RepetitionCheck()
    result = check.run([])
    assert result["score"] == 1.0
    assert result["status"] == "ok"
