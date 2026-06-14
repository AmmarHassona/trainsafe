"""End-to-end integration test: TrainSafeCallback with a real Trainer + real model.

Uses a tiny GPT-2 model created from scratch (no downloads required) and a synthetic
dataset of random token IDs. Verifies that:
  - on_evaluate fires and receives model/processing_class/eval_dataloader in kwargs
  - sample collection, generation, and checks all run without crashing
  - the callback stops training when health is below stop_threshold
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from trainsafe import TrainSafeCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_tokenizer(vocab_size: int = 100) -> PreTrainedTokenizerFast:
    """Build a minimal whitespace tokenizer over token-id strings 0..vocab_size-1."""
    vocab = {str(i): i for i in range(vocab_size)}
    vocab["[UNK]"] = vocab_size
    vocab["[PAD]"] = vocab_size + 1
    vocab["[EOS]"] = vocab_size + 2
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    return fast


def _make_tiny_model(vocab_size: int = 103) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=64,
        vocab_size=vocab_size,
        bos_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
    )
    return GPT2LMHeadModel(cfg)


def _make_sft_dataset(n: int = 20, seq_len: int = 32, vocab_size: int = 100) -> Dataset:
    """Synthetic SFT dataset: random input_ids with -100 labels for prompt tokens.

    Uses default_data_collator (not DataCollatorForLanguageModeling) so our
    custom labels are preserved as-is through the DataLoader.
    """
    input_ids = torch.randint(1, vocab_size, (n, seq_len)).tolist()
    # First half is prompt (-100 mask), second half is completion (real token ids).
    labels = [
        [-100] * (seq_len // 2) + row[seq_len // 2:]
        for row in input_ids
    ]
    attention_mask = [[1] * seq_len for _ in range(n)]
    return Dataset.from_dict({
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    })


# ---------------------------------------------------------------------------
# Spy callback
# ---------------------------------------------------------------------------


class SpyCallback(TrainerCallback):
    def __init__(self):
        self.evaluate_calls: list[dict] = []

    def on_evaluate(self, args, state, control, **kwargs):
        self.evaluate_calls.append({
            "step": state.global_step,
            "has_model": "model" in kwargs,
            "has_processing_class": "processing_class" in kwargs or "tokenizer" in kwargs,
            "has_eval_dataloader": "eval_dataloader" in kwargs,
        })


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_setup():
    tokenizer = _make_tiny_tokenizer()
    model = _make_tiny_model(vocab_size=len(tokenizer))
    train_ds = _make_sft_dataset(n=20, seq_len=32, vocab_size=100)
    eval_ds = _make_sft_dataset(n=8, seq_len=32, vocab_size=100)
    return model, tokenizer, train_ds, eval_ds


def _base_args(output_dir: str, **overrides) -> TrainingArguments:
    """Shared TrainingArguments for integration tests."""
    kwargs = dict(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=2,
        max_steps=3,  # odd number so final step ≠ eval step
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        report_to="none",
        use_cpu=True,
    )
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_callback_kwargs_available(tiny_setup):
    """Verify Trainer passes model, processing_class, and eval_dataloader to on_evaluate."""
    model, tokenizer, train_ds, eval_ds = tiny_setup
    spy = SpyCallback()

    trainer = Trainer(
        model=model,
        args=_base_args("/tmp/trainsafe_test_kwargs"),
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[spy],
    )
    trainer.train()

    assert len(spy.evaluate_calls) >= 1, "on_evaluate was never called"
    call = spy.evaluate_calls[0]
    assert call["has_model"], "model not in on_evaluate kwargs"
    assert call["has_processing_class"], "processing_class/tokenizer not in on_evaluate kwargs"
    assert call["has_eval_dataloader"], "eval_dataloader not in on_evaluate kwargs"


def test_trainsafe_callback_does_not_crash(tiny_setup):
    """TrainSafeCallback runs through a full eval loop without raising."""
    model, tokenizer, train_ds, eval_ds = tiny_setup
    callback = TrainSafeCallback(
        warn_threshold=0.0,
        stop_threshold=0.0,
        num_inference_samples=3,
        max_new_tokens=8,
        log_to_wandb=False,
    )

    trainer = Trainer(
        model=model,
        args=_base_args("/tmp/trainsafe_test_nocrash"),
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[callback],
    )
    trainer.train()  # must not raise


def test_trainsafe_stops_training_when_threshold_exceeded(tiny_setup):
    """TrainSafeCallback stops training when stop_threshold is set impossibly high."""
    model, tokenizer, train_ds, eval_ds = tiny_setup

    stopped_steps: list[int] = []

    class StepSpy(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if control.should_training_stop:
                stopped_steps.append(state.global_step)

    # stop_threshold=1.1 is impossible to beat — any health ≤ 1.0 stops training
    callback = TrainSafeCallback(
        warn_threshold=1.1,
        stop_threshold=1.1,
        num_inference_samples=3,
        max_new_tokens=8,
        log_to_wandb=False,
    )

    trainer = Trainer(
        model=model,
        args=_base_args("/tmp/trainsafe_test_stop", max_steps=10),
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[callback, StepSpy()],
    )
    trainer.train()

    # Training should have been cut short by our callback
    assert trainer.state.global_step < 10, (
        f"Training ran to completion (step {trainer.state.global_step}); "
        "expected TrainSafeCallback to stop it early"
    )
