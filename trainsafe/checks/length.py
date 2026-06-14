from __future__ import annotations

_COLLAPSE_RATIO = 0.50   # warn if avg drops below 50% of baseline
_SPIKE_RATIO = 3.0       # warn if avg rises above 300% of baseline


class LengthCheck:
    """Track output length distribution and flag collapses or spikes."""

    wandb_key = "trainsafe/avg_output_length"

    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.baseline_mean: float | None = None
        self._ema_alpha = ema_alpha

    def reset(self) -> None:
        self.baseline_mean = None

    def run(self, outputs: list[str]) -> dict:
        if not outputs:
            return _skip()

        # word count as a proxy for token count — fast and model-agnostic
        lengths = [len(o.split()) for o in outputs]
        mean = sum(lengths) / len(lengths)

        if self.baseline_mean is None:
            self.baseline_mean = mean
            return _ok(mean, f"Output length normal (avg {mean:.0f} words)")

        if self.baseline_mean == 0:
            return _ok(mean, f"Output length normal (avg {mean:.0f} words)")

        ratio = mean / self.baseline_mean

        if ratio < _COLLAPSE_RATIO:
            return _warn(
                mean,
                f"Output length collapsed (avg {mean:.0f} words vs baseline {self.baseline_mean:.0f})",
            )

        if ratio > _SPIKE_RATIO:
            return _warn(
                mean,
                f"Output length spike (avg {mean:.0f} words vs baseline {self.baseline_mean:.0f})",
            )

        if self._ema_alpha > 0:
            self.baseline_mean = (1 - self._ema_alpha) * self.baseline_mean + self._ema_alpha * mean

        return _ok(mean, f"Output length normal (avg {mean:.0f} words)")


def _skip() -> dict:
    return {"score": 1.0, "message": None, "status": "skip", "wandb_key": "trainsafe/avg_output_length", "wandb_value": 0.0}


def _ok(mean: float, message: str) -> dict:
    return {"score": 1.0, "message": message, "status": "ok", "wandb_key": "trainsafe/avg_output_length", "wandb_value": mean}


def _warn(mean: float, message: str) -> dict:
    return {"score": 0.0, "message": message, "status": "warn", "wandb_key": "trainsafe/avg_output_length", "wandb_value": mean}
