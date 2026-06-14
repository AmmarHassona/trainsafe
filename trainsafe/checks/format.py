from __future__ import annotations

import json
import re
from collections import Counter

_MARKDOWN_PATTERN = re.compile(
    r"(^#{1,6} |\*\*|__|\[.+\]\(.+\)|^[-*] )",
    re.MULTILINE,
)


def detect_format(text: str) -> str:
    text = text.strip()
    if not text:
        return "plain"
    try:
        json.loads(text)
        return "json"
    except (json.JSONDecodeError, ValueError):
        pass
    if _MARKDOWN_PATTERN.search(text):
        return "markdown"
    return "plain"


class FormatCheck:
    """Detect format drift between baseline and current checkpoint."""

    wandb_key = "trainsafe/format_consistency"

    def __init__(self, baseline_grace: int = 3) -> None:
        self.baseline_format: str | None = None
        self._grace = baseline_grace
        self._pending_format: str | None = None
        self._pending_count: int = 0

    def reset(self) -> None:
        self.baseline_format = None
        self._pending_format = None
        self._pending_count = 0

    def run(self, outputs: list[str]) -> dict:
        non_empty = [o for o in outputs if o.strip()]
        if not non_empty:
            return _skip()

        formats = [detect_format(o) for o in non_empty]
        fmt = Counter(formats).most_common(1)[0][0]

        if self.baseline_format is None:
            self.baseline_format = fmt
            return _result(1.0, f"Format consistent ({fmt})", "ok", 1.0)

        if fmt == self.baseline_format:
            self._pending_format = None
            self._pending_count = 0
            return _result(1.0, f"Format consistent ({fmt})", "ok", 1.0)

        if fmt == self._pending_format:
            self._pending_count += 1
        else:
            self._pending_format = fmt
            self._pending_count = 1

        if self._pending_count >= self._grace:
            self.baseline_format = fmt
            self._pending_format = None
            self._pending_count = 0
            return _result(1.0, f"Format baseline updated to {fmt}", "ok", 1.0)

        return _result(
            0.0,
            f"Format drift — was {self.baseline_format}, now {fmt}",
            "warn",
            0.0,
        )


def _skip() -> dict:
    return {"score": 1.0, "message": None, "status": "skip", "wandb_key": "trainsafe/format_consistency", "wandb_value": 1.0}


def _result(score: float, message: str, status: str, wandb_value: float) -> dict:
    return {"score": score, "message": message, "status": status, "wandb_key": "trainsafe/format_consistency", "wandb_value": wandb_value}
