import pytest
from trainsafe.checks.echo import EchoCheck, _is_echo


def test_no_echo_passes():
    check = EchoCheck()
    samples = [
        {"prompt": "What is the capital of France?", "output": "Paris is the capital city of France."},
        {"prompt": "Explain machine learning briefly.", "output": "Machine learning teaches computers to learn from data."},
    ]
    result = check.run(samples)
    assert result["score"] == 1.0
    assert result["status"] == "ok"


def test_echo_detected():
    check = EchoCheck()
    prompt = "What is the capital of France and what is its population size today?"
    samples = [{"prompt": prompt, "output": prompt}] * 5  # output == prompt
    result = check.run(samples)
    assert result["score"] < 1.0
    assert result["status"] == "warn"
    assert "echoing" in result["message"].lower()


def test_partial_echo_partial_score():
    check = EchoCheck()
    prompt = "What is the capital of France and what is its population size today?"
    samples = [
        {"prompt": prompt, "output": prompt},  # echo
        {"prompt": prompt, "output": "Paris is the capital of France with 2 million people."},  # not echo
    ]
    result = check.run(samples)
    # echo_rate = 0.5
    assert result["score"] == pytest.approx(0.5)


def test_empty_samples():
    check = EchoCheck()
    result = check.run([])
    assert result["score"] == 1.0


def test_is_echo_helper_exact():
    prompt = "explain machine learning in simple terms for beginners"
    assert _is_echo(prompt, prompt)


def test_is_echo_helper_distinct():
    assert not _is_echo(
        "What is the weather like today in Berlin?",
        "It is sunny and warm with clear skies.",
    )


def test_echo_factual_qa_not_flagged():
    """A factual answer containing prompt words should NOT be an echo."""
    check = EchoCheck()
    samples = [
        {"prompt": "What is the capital of Germany?", "output": "Berlin is the capital of Germany."},
        {"prompt": "What is the largest planet?", "output": "Jupiter is the largest planet in the solar system."},
    ]
    result = check.run(samples)
    assert result["score"] == 1.0
    assert result["status"] == "ok"


def test_echo_verbatim_copy_is_flagged():
    check = EchoCheck()
    samples = [{"prompt": "Tell me about machine learning algorithms", "output": "Tell me about machine learning algorithms and how they work"}]
    result = check.run(samples)
    assert result["score"] < 1.0
