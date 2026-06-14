from __future__ import annotations

try:
    import wandb as _wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def log(
    step: int,
    results: list[dict],
    overall_health: float,
    probe_pass_rate: float | None = None,
) -> None:
    """Log trainsafe metrics to W&B. No-ops silently if wandb is not installed or not initialized."""
    if not _HAS_WANDB:
        return

    run = getattr(_wandb, "run", None)
    if run is None:
        return

    metrics: dict[str, float] = {}
    for r in results:
        key = r.get("wandb_key")
        if key and r["status"] != "skip":
            metrics[key] = r["wandb_value"]

    if probe_pass_rate is not None:
        metrics["trainsafe/custom_probe_pass_rate"] = probe_pass_rate

    metrics["trainsafe/overall_health"] = overall_health

    _wandb.log(metrics, step=step)
