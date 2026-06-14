from __future__ import annotations

_NGRAM_SIZE = 4
_REPEATED_NGRAM_RATIO = 0.30   # fraction of n-grams that are duplicates to flag one output
_SAMPLE_WARN_RATIO = 0.50      # fraction of samples with repetition to warn


class RepetitionCheck:
    """Detect n-gram repetition loops within generated outputs."""

    wandb_key = "trainsafe/repetition_rate"

    def run(self, outputs: list[str]) -> dict:
        if not outputs:
            return {"score": 1.0, "message": "No repetition detected", "status": "ok", "wandb_key": self.wandb_key, "wandb_value": 0.0}

        repeated = [o for o in outputs if _has_repetition(o)]
        rate = len(repeated) / len(outputs)
        status = "warn" if rate > _SAMPLE_WARN_RATIO else "ok"
        message = (
            f"Repetition detected in {len(repeated)}/{len(outputs)} outputs"
            if repeated
            else "No repetition detected"
        )

        return {
            "score": 1.0 - rate,
            "message": message,
            "status": status,
            "wandb_key": self.wandb_key,
            "wandb_value": rate,
        }


def _has_repetition(text: str, n: int = _NGRAM_SIZE, threshold: float = _REPEATED_NGRAM_RATIO) -> bool:
    words = text.split()
    if len(words) < n * 2:
        return False

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return False

    # Repetitiveness = fraction of n-grams that are duplicates of an earlier one.
    # A highly repetitive loop has few unique n-grams relative to the total,
    # so (1 - unique/total) trends toward 1.0.
    unique_count = len(set(ngrams))
    repetitiveness = 1.0 - (unique_count / len(ngrams))
    return repetitiveness > threshold
