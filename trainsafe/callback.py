from __future__ import annotations

import warnings
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from trainsafe.checks import EchoCheck, FormatCheck, LanguageCheck, LengthCheck, RepetitionCheck
from trainsafe.reporters import terminal as terminal_reporter
from trainsafe.reporters import wandb as wandb_reporter
from trainsafe.utils import extract_prompt_ids, generate_output, sample_from_dataloader


class TrainSafeCallback(TrainerCallback):
    """Behavioral health checks at each eval checkpoint.

    Works with ``transformers.Trainer``, ``trl.SFTTrainer``,
    ``trl.DPOTrainer``, and ``trl.GRPOTrainer``.
    """

    def __init__(
        self,
        probes: str | None = None,
        warn_threshold: float = 0.7,
        stop_threshold: float = 0.4,
        probe_every_n_steps: int | None = None,
        num_inference_samples: int = 5,
        max_new_tokens: int = 256,
        log_to_wandb: bool = True,
    ) -> None:
        if probe_every_n_steps is not None and probe_every_n_steps <= 0:
            raise ValueError("probe_every_n_steps must be a positive integer or None")

        self.warn_threshold = warn_threshold
        self.stop_threshold = stop_threshold
        self.probe_every_n_steps = probe_every_n_steps
        self.num_inference_samples = num_inference_samples
        self.max_new_tokens = max_new_tokens
        self.log_to_wandb = log_to_wandb

        self._language_check = LanguageCheck()
        self._length_check = LengthCheck()
        self._repetition_check = RepetitionCheck()
        self._echo_check = EchoCheck()
        self._format_check = FormatCheck()

        self._probe_runner = None
        if probes is not None:
            from trainsafe.probes import ProbeRunner

            self._probe_runner = ProbeRunner(probes, max_new_tokens=self.max_new_tokens)

        self._best_health: float = 0.0
        self._best_step: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._language_check.reset()
        self._length_check.reset()
        self._format_check.reset()
        self._best_health = 0.0
        self._best_step = 0

        if self.probe_every_n_steps is not None:
            eval_strategy = getattr(args, "eval_strategy", None)
            strategy_value = eval_strategy.value if hasattr(eval_strategy, "value") else str(eval_strategy)
            if strategy_value == "steps":
                eval_steps = int(getattr(args, "eval_steps", 0))
                if eval_steps > 0 and self.probe_every_n_steps % eval_steps != 0:
                    warnings.warn(
                        f"TrainSafe: probe_every_n_steps={self.probe_every_n_steps} is not a multiple of "
                        f"eval_steps={eval_steps}. Checks will fire at irregular intervals. "
                        f"Set probe_every_n_steps to a multiple of eval_steps (e.g. {eval_steps * round(self.probe_every_n_steps / eval_steps)})."
                    )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step

        if self.probe_every_n_steps is not None and step % self.probe_every_n_steps != 0:
            return

        model = kwargs.get("model")
        # transformers >= 4.45 passes processing_class; older versions pass tokenizer.
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        eval_dataloader = kwargs.get("eval_dataloader")

        if model is None or tokenizer is None:
            warnings.warn("TrainSafe: model or tokenizer not available — skipping checks.")
            return

        if eval_dataloader is None:
            warnings.warn("TrainSafe: eval_dataloader not available — skipping checks.")
            return

        samples = self._collect_samples(model, tokenizer, eval_dataloader)

        if not samples:
            warnings.warn("TrainSafe: inference produced no samples — skipping checks.")
            return

        outputs = [s["output"] for s in samples]
        prompt_output_pairs = [{"prompt": s["prompt"], "output": s["output"]} for s in samples]

        results = [
            self._language_check.run(outputs),
            self._length_check.run(outputs),
            self._repetition_check.run(outputs),
            self._echo_check.run(prompt_output_pairs),
            self._format_check.run(outputs),
        ]

        probe_pass_rate: float | None = None
        if self._probe_runner is not None:
            try:
                probe_result = self._probe_runner.run(model, tokenizer)
                probe_pass_rate = probe_result["pass_rate"]
                results.append(
                    {
                        "score": probe_pass_rate,
                        "message": f"Custom probes: {probe_pass_rate:.0%} passed",
                        "status": "ok" if probe_pass_rate >= self.warn_threshold else "warn",
                        "wandb_key": "trainsafe/custom_probe_pass_rate",
                        "wandb_value": probe_pass_rate,
                    }
                )
            except Exception as exc:
                warnings.warn(f"TrainSafe: probe runner failed — {exc}")

        scoreable = [r for r in results if r["status"] != "skip"]
        overall_health = sum(r["score"] for r in scoreable) / len(scoreable) if scoreable else 1.0

        if overall_health > self._best_health:
            self._best_health = overall_health
            self._best_step = step

        stopped = overall_health < self.stop_threshold
        terminal_reporter.report(
            step,
            results,
            overall_health,
            stopped=stopped,
            checkpoint_hint=_checkpoint_hint(args, self._best_step) if stopped else None,
        )

        if self.log_to_wandb:
            wandb_reporter.log(step, results, overall_health, probe_pass_rate)

        if overall_health < self.stop_threshold:
            control.should_training_stop = True
        elif overall_health < self.warn_threshold:
            warnings.warn(
                f"TrainSafe: overall health {overall_health:.2f} below warn threshold {self.warn_threshold}"
            )

    def _collect_samples(
        self,
        model: torch.nn.Module,
        tokenizer,
        eval_dataloader,
    ) -> list[dict]:
        """Run inference on sampled eval examples; return list of {prompt, output}."""
        raw = sample_from_dataloader(eval_dataloader, self.num_inference_samples)
        samples: list[dict] = []

        was_training = model.training
        model.eval()

        try:
            for item in raw:
                try:
                    if "prompt_text" in item:
                        # GRPO-style: raw string or conversational list-of-dicts.
                        prompt = item["prompt_text"]
                        if isinstance(prompt, (list, tuple)):
                            # Conversational format — apply chat template if available.
                            if hasattr(tokenizer, "apply_chat_template"):
                                prompt_str = tokenizer.apply_chat_template(
                                    prompt,
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                            else:
                                prompt_str = " ".join(
                                    m.get("content", "") for m in prompt if isinstance(m, dict)
                                )
                        else:
                            prompt_str = prompt

                        encoded = tokenizer(prompt_str, return_tensors="pt")
                        prompt_ids = encoded["input_ids"][0]
                    else:
                        # SFT / DPO-style: pre-tokenized batch item.
                        prompt_ids = extract_prompt_ids(
                            item["input_ids"],
                            labels=item.get("labels"),
                            completion_mask=item.get("completion_mask"),
                        )
                        if len(prompt_ids) == 0:
                            warnings.warn(
                                "TrainSafe: extracted empty prompt from batch item — skipping."
                            )
                            continue
                        prompt_str = tokenizer.decode(prompt_ids, skip_special_tokens=True)

                    output_text = generate_output(
                        model, tokenizer, prompt_ids, self.max_new_tokens
                    )
                    samples.append({"prompt": prompt_str, "output": output_text})
                except Exception as exc:
                    warnings.warn(f"TrainSafe: inference failed on one sample — {exc}")
        finally:
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()

        return samples


def _checkpoint_hint(args: TrainingArguments, best_step: int) -> str | None:
    """Return a step string if a checkpoint plausibly exists there, else None."""
    raw = getattr(args, "save_strategy", "no")
    save_strategy = raw.value if hasattr(raw, "value") else str(raw)
    if save_strategy == "no" or best_step == 0:
        return None
    if save_strategy == "steps":
        save_steps = int(getattr(args, "save_steps", 0))
        if save_steps > 0:
            aligned = (best_step // save_steps) * save_steps
            return str(aligned) if aligned > 0 else None
    return str(best_step)
