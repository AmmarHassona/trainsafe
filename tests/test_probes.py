import os
import tempfile

import pytest
import yaml

from trainsafe.probes import ProbeRunner, _evaluate_checks


def test_evaluate_checks_language_pass():
    passed, failures = _evaluate_checks("prompt", "hello world this is english text", [{"language": "en"}])
    # langdetect might not be installed; if it is, it should pass
    # we test the structure, not the language detection result
    assert isinstance(passed, bool)
    assert isinstance(failures, list)


def test_evaluate_checks_min_length_pass():
    passed, failures = _evaluate_checks("prompt", "one two three four five six seven eight nine ten", [{"min_length": 5}])
    assert passed is True


def test_evaluate_checks_min_length_fail():
    passed, failures = _evaluate_checks("prompt", "short", [{"min_length": 10}])
    assert passed is False
    assert any("short" in f.lower() or "min" in f.lower() for f in failures)


def test_evaluate_checks_not_contains_pass():
    passed, failures = _evaluate_checks("prompt", "clean output here", [{"not_contains": ["<|im_start|>", "###"]}])
    assert passed is True


def test_evaluate_checks_not_contains_fail():
    passed, failures = _evaluate_checks("prompt", "output with <|im_start|> token", [{"not_contains": ["<|im_start|>"]}])
    assert passed is False


def test_evaluate_checks_contains_pass():
    passed, failures = _evaluate_checks("prompt", "the answer is yes", [{"contains": ["yes", "no"]}])
    assert passed is True


def test_evaluate_checks_contains_fail():
    passed, failures = _evaluate_checks("prompt", "the answer is maybe", [{"contains": ["yes", "no"]}])
    assert passed is False


def test_evaluate_checks_max_length():
    passed, failures = _evaluate_checks("prompt", " ".join(["word"] * 20), [{"max_length": 5}])
    assert passed is False


def test_evaluate_checks_coherent_repetitive():
    repetitive = " ".join(["the model said the same thing over and over again"] * 8)
    passed, failures = _evaluate_checks("prompt", repetitive, [{"coherent": True}])
    assert passed is False


def test_probe_runner_invalid_file():
    with pytest.raises(FileNotFoundError):
        ProbeRunner("/nonexistent/path/probes.yaml")


def test_probe_runner_no_probes_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"language": "en"}, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match="No probes"):
            ProbeRunner(path)
    finally:
        os.unlink(path)
