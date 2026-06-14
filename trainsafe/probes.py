from __future__ import annotations

import torch
import yaml

from trainsafe.checks.format import detect_format
from trainsafe.checks.repetition import _has_repetition
from trainsafe.utils import generate_output

try:
    from langdetect import LangDetectException, detect as _detect

    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False


class ProbeRunner:
    """Run custom YAML-defined probes against the current model checkpoint."""

    def __init__(self, probe_file: str, max_new_tokens: int = 256) -> None:
        with open(probe_file) as f:
            config = yaml.safe_load(f)
        self._probes: list[dict] = config.get("probes", [])
        if not self._probes:
            raise ValueError(f"No probes found in {probe_file}")
        self._max_new_tokens = max_new_tokens

    def run(self, model: torch.nn.Module, tokenizer) -> dict:
        """Return {'pass_rate': float, 'results': list[dict]}."""
        results = []
        for probe in self._probes:
            prompt = probe["prompt"]
            checks = probe.get("checks", [])
            output = generate_output(model, tokenizer, _encode(tokenizer, prompt), self._max_new_tokens)
            passed, failures = _evaluate_checks(prompt, output, checks)
            results.append(
                {
                    "prompt": prompt,
                    "output": output,
                    "passed": passed,
                    "failures": failures,
                }
            )

        pass_count = sum(1 for r in results if r["passed"])
        pass_rate = pass_count / len(results) if results else 1.0
        return {"pass_rate": pass_rate, "results": results}


def _encode(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt")["input_ids"][0]


def _evaluate_checks(prompt: str, output: str, checks: list[dict]) -> tuple[bool, list[str]]:
    failures: list[str] = []

    for check in checks:
        for check_type, value in check.items():
            failure = _run_check(check_type, value, prompt, output)
            if failure:
                failures.append(failure)

    return len(failures) == 0, failures


def _run_check(check_type: str, value, prompt: str, output: str) -> str | None:
    """Return a failure message string if the check fails, else None."""
    if check_type == "language":
        if not _HAS_LANGDETECT:
            return None  # skip silently
        try:
            detected = _detect(output.strip()) if output.strip() else None
        except LangDetectException:
            detected = None
        if detected and detected != value:
            return f"Expected language {value!r}, got {detected!r}"

    elif check_type == "min_length":
        word_count = len(output.split())
        if word_count < int(value):
            return f"Output too short: {word_count} words (min {value})"

    elif check_type == "max_length":
        word_count = len(output.split())
        if word_count > int(value):
            return f"Output too long: {word_count} words (max {value})"

    elif check_type == "not_contains":
        for forbidden in value:
            if forbidden in output:
                return f"Output contains forbidden string: {forbidden!r}"

    elif check_type == "contains":
        if not any(required in output for required in value):
            return f"Output missing required strings: {value}"

    elif check_type == "format":
        detected_fmt = detect_format(output)
        if detected_fmt != value:
            return f"Expected format {value!r}, got {detected_fmt!r}"

    elif check_type == "coherent":
        if value:
            if not output.strip() or len(output.split()) < 3:
                return "Output is incoherent (empty or too short)"
            if _has_repetition(output):
                return "Output is incoherent (repetitive n-grams detected)"

    return None
