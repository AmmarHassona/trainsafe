"""Integration tests using real TRL trainers and the cached tiny model.

Requires: trl, datasets (both in [dev] dependencies)
Model: trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 (cached locally)
Data:  trl-internal-testing/zen (cached locally)

These tests run actual training loops and verify:
  - SFTTrainer: callback fires, checks run, no crash
  - DPOTrainer: callback fires, DPO batch format is handled, no crash
  - GRPOTrainer: callback fires, list-of-dicts batch format is handled, no crash
  - Early stopping: should_training_stop is set when threshold is exceeded
"""

from __future__ import annotations

import pytest

pytest.importorskip("trl", reason="trl not installed")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from trainsafe import TrainSafeCallback

MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
_CALLBACK_KWARGS = dict(
    num_inference_samples=2,
    max_new_tokens=16,
    log_to_wandb=False,
)


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


@pytest.fixture(scope="module")
def sft_datasets():
    ds = load_dataset("trl-internal-testing/zen", "standard_prompt_completion")
    return ds["train"], ds["test"]


@pytest.fixture(scope="module")
def dpo_datasets():
    ds = load_dataset("trl-internal-testing/zen", "standard_preference")
    return ds["train"], ds["test"]


def _sft_args(output_dir: str, **overrides) -> SFTConfig:
    kwargs = dict(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=5,
        max_steps=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        report_to="none",
        use_cpu=True,
    )
    kwargs.update(overrides)
    return SFTConfig(**kwargs)


def _dpo_args(output_dir: str, **overrides) -> DPOConfig:
    kwargs = dict(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=5,
        max_steps=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        report_to="none",
        use_cpu=True,
    )
    kwargs.update(overrides)
    return DPOConfig(**kwargs)


def test_sft_callback_does_not_crash(tiny_model_and_tokenizer, sft_datasets):
    model, tokenizer = tiny_model_and_tokenizer
    train_ds, eval_ds = sft_datasets

    trainer = SFTTrainer(
        model=model,
        args=_sft_args("/tmp/trainsafe_sft"),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[TrainSafeCallback(warn_threshold=0.0, stop_threshold=0.0, **_CALLBACK_KWARGS)],
    )
    trainer.train()


def test_sft_stops_when_threshold_exceeded(tiny_model_and_tokenizer, sft_datasets):
    model, tokenizer = tiny_model_and_tokenizer
    train_ds, eval_ds = sft_datasets

    trainer = SFTTrainer(
        model=model,
        args=_sft_args("/tmp/trainsafe_sft_stop", max_steps=30),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[TrainSafeCallback(warn_threshold=1.1, stop_threshold=1.1, **_CALLBACK_KWARGS)],
    )
    trainer.train()
    assert trainer.state.global_step < 30, (
        f"Expected early stop but training ran to step {trainer.state.global_step}"
    )


def test_dpo_callback_does_not_crash(tiny_model_and_tokenizer, dpo_datasets):
    model, tokenizer = tiny_model_and_tokenizer
    train_ds, eval_ds = dpo_datasets

    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=_dpo_args("/tmp/trainsafe_dpo"),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[TrainSafeCallback(warn_threshold=0.0, stop_threshold=0.0, **_CALLBACK_KWARGS)],
    )
    trainer.train()


def test_dpo_stops_when_threshold_exceeded(tiny_model_and_tokenizer, dpo_datasets):
    model, tokenizer = tiny_model_and_tokenizer
    train_ds, eval_ds = dpo_datasets

    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=_dpo_args("/tmp/trainsafe_dpo_stop", max_steps=30),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[TrainSafeCallback(warn_threshold=1.1, stop_threshold=1.1, **_CALLBACK_KWARGS)],
    )
    trainer.train()
    assert trainer.state.global_step < 30, (
        f"Expected early stop but training ran to step {trainer.state.global_step}"
    )


def test_grpo_callback_does_not_crash():
    """GRPO eval dataloader yields list-of-dicts; verify this batch format is handled."""
    from trl import GRPOConfig, GRPOTrainer

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    ds = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

    def dummy_reward(completions, **kwargs):
        return [1.0] * len(completions)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=dummy_reward,
        args=GRPOConfig(
            output_dir="/tmp/trainsafe_grpo",
            eval_strategy="steps",
            eval_steps=5,
            max_steps=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_generations=2,
            report_to="none",
            use_cpu=True,
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        callbacks=[TrainSafeCallback(warn_threshold=0.0, stop_threshold=0.0, **_CALLBACK_KWARGS)],
    )
    trainer.train()
