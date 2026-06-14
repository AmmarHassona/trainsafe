from __future__ import annotations

_OVERLAP_THRESHOLD = 0.80   # fraction of prompt content words found in output to flag as echo

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "up", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "whom", "not", "no", "or", "and",
    "but", "if", "then", "so", "than", "too", "very", "just", "also",
})


def _content_words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w not in _STOP_WORDS and w.isalpha()}


class EchoCheck:
    """Detect outputs that are copying the prompt verbatim."""

    wandb_key = "trainsafe/echo_rate"

    def run(self, samples: list[dict]) -> dict:
        """
        Args:
            samples: list of {'prompt': str, 'output': str}
        """
        if not samples:
            return _ok(0.0)

        echo_count = sum(1 for s in samples if _is_echo(s["prompt"], s["output"]))
        rate = echo_count / len(samples)
        status = "warn" if echo_count > 0 else "ok"
        message = (
            f"Model echoing prompt in {echo_count}/{len(samples)} outputs"
            if echo_count > 0
            else "No prompt echoing"
        )

        return {
            "score": 1.0 - rate,
            "message": message,
            "status": status,
            "wandb_key": "trainsafe/echo_rate",
            "wandb_value": rate,
        }


def _is_echo(prompt: str, output: str) -> bool:
    prompt_words = _content_words(prompt)
    output_words = _content_words(output)
    if len(prompt_words) < 3 or not output_words:
        return False
    overlap = len(prompt_words & output_words) / len(prompt_words)
    return overlap > _OVERLAP_THRESHOLD


def _ok(rate: float) -> dict:
    return {"score": 1.0 - rate, "message": "No prompt echoing", "status": "ok", "wandb_key": "trainsafe/echo_rate", "wandb_value": rate}
