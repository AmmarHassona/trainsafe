from __future__ import annotations

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

_GREEN = Fore.GREEN if _HAS_COLORAMA else ""
_YELLOW = Fore.YELLOW if _HAS_COLORAMA else ""
_RED = Fore.RED if _HAS_COLORAMA else ""
_RESET = Style.RESET_ALL if _HAS_COLORAMA else ""


def report(
    step: int,
    results: list[dict],
    overall_health: float,
    stopped: bool = False,
    checkpoint_hint: str | None = None,
) -> None:
    prefix = f"[TrainSafe @ step {step}]"
    for r in results:
        if r["status"] in ("skip", None) or not r.get("message"):
            continue
        icon, color = _icon_color(r["status"])
        print(f"{color}{prefix} {icon} {r['message']}{_RESET}")

    health_color = _health_color(overall_health)
    print(f"{health_color}{prefix} Overall health: {overall_health:.2f}{_RESET}")

    if stopped:
        if checkpoint_hint:
            rec = f" Recommended checkpoint: step {checkpoint_hint}."
        else:
            rec = " No checkpoints were saved (save_strategy='no') — re-run with save_strategy='steps' to preserve intermediate checkpoints."
        print(f"{_RED}>>> TrainSafe stopped training.{rec}{_RESET}")


def _icon_color(status: str) -> tuple[str, str]:
    if status == "fail":
        return "🚨", _RED
    if status == "warn":
        return "⚠️ ", _YELLOW
    return "✅", _GREEN


def _health_color(health: float) -> str:
    if health >= 0.7:
        return _GREEN
    if health >= 0.4:
        return _YELLOW
    return _RED
