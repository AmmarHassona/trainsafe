from __future__ import annotations

from collections import Counter

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic results
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

_MIN_CHARS = 20  # shorter outputs produce unreliable language detection


class LanguageCheck:
    """Detect language drift between baseline and current checkpoint."""

    wandb_key = "trainsafe/language_consistency"

    def __init__(self, baseline_grace: int = 3) -> None:
        self.baseline_language: str | None = None
        self._grace = baseline_grace
        self._pending_language: str | None = None
        self._pending_count: int = 0

    def reset(self) -> None:
        self.baseline_language = None
        self._pending_language = None
        self._pending_count = 0

    def run(self, outputs: list[str]) -> dict:
        if not _HAS_LANGDETECT:
            return _skip("Language check skipped — langdetect not installed")

        detectable = [o for o in outputs if len(o.strip()) >= _MIN_CHARS]
        if not detectable:
            return _skip("Language check skipped — outputs too short to detect")

        detected: list[str] = []
        for text in detectable:
            try:
                detected.append(detect(text))
            except LangDetectException:
                pass

        if not detected:
            return _skip("Language check skipped — detection failed on all outputs")

        lang = Counter(detected).most_common(1)[0][0]

        if self.baseline_language is None:
            self.baseline_language = lang
            return _result(1.0, f"Language consistent ({lang})", "ok", self.wandb_key, 1.0)

        if lang == self.baseline_language:
            self._pending_language = None
            self._pending_count = 0
            return _result(1.0, f"Language consistent ({lang})", "ok", self.wandb_key, 1.0)

        if lang == self._pending_language:
            self._pending_count += 1
        else:
            self._pending_language = lang
            self._pending_count = 1

        if self._pending_count >= self._grace:
            self.baseline_language = lang
            self._pending_language = None
            self._pending_count = 0
            return _result(1.0, f"Language baseline updated to {lang}", "ok", self.wandb_key, 1.0)

        return _result(
            0.0,
            f"Language drift — expected {self.baseline_language}, got {lang}",
            "fail",
            self.wandb_key,
            0.0,
        )


def _skip(message: str) -> dict:
    return _result(1.0, message, "skip", "trainsafe/language_consistency", 1.0)


def _result(score: float, message: str, status: str, wandb_key: str, wandb_value: float) -> dict:
    return {
        "score": score,
        "message": message,
        "status": status,
        "wandb_key": wandb_key,
        "wandb_value": wandb_value,
    }
