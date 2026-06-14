"""Tests for trainsafe.utils — extract_prompt_ids and sample_from_dataloader."""

import pytest
import torch

from trainsafe.utils import extract_prompt_ids, sample_from_dataloader


# ---------------------------------------------------------------------------
# extract_prompt_ids
# ---------------------------------------------------------------------------


def test_sft_labels_extracts_prompt():
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    labels = torch.tensor([-100, -100, -100, 10, 11])
    result = extract_prompt_ids(input_ids, labels=labels)
    assert result.tolist() == [1, 2, 3]


def test_sft_labels_all_masked_returns_full():
    """All -100 means the whole sequence is the prompt."""
    input_ids = torch.tensor([1, 2, 3])
    labels = torch.tensor([-100, -100, -100])
    result = extract_prompt_ids(input_ids, labels=labels)
    assert result.tolist() == [1, 2, 3]


def test_sft_labels_first_token_completion_returns_empty():
    """Bug fix: i==0 should return empty tensor, not the full input."""
    input_ids = torch.tensor([1, 2, 3])
    labels = torch.tensor([10, 11, 12])  # first token already a completion
    result = extract_prompt_ids(input_ids, labels=labels)
    assert result.tolist() == []


def test_dpo_completion_mask_extracts_prompt():
    input_ids = torch.tensor([7, 8, 9, 10, 0])
    completion_mask = torch.tensor([0, 0, 1, 1, 0])  # trailing 0 is padding
    result = extract_prompt_ids(input_ids, completion_mask=completion_mask)
    assert result.tolist() == [7, 8]


def test_dpo_completion_mask_no_completion_returns_full():
    input_ids = torch.tensor([1, 2, 3])
    completion_mask = torch.tensor([0, 0, 0])
    result = extract_prompt_ids(input_ids, completion_mask=completion_mask)
    assert result.tolist() == [1, 2, 3]


def test_dpo_completion_mask_takes_priority_over_labels():
    """completion_mask should be used when both are provided."""
    input_ids = torch.tensor([1, 2, 3, 4])
    completion_mask = torch.tensor([0, 1, 1, 1])
    labels = torch.tensor([-100, -100, 10, 11])
    result = extract_prompt_ids(input_ids, labels=labels, completion_mask=completion_mask)
    assert result.tolist() == [1]


def test_no_mask_returns_full():
    input_ids = torch.tensor([1, 2, 3])
    result = extract_prompt_ids(input_ids)
    assert result.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# sample_from_dataloader — SFT format
# ---------------------------------------------------------------------------


def _make_sft_batch(batch_size: int, seq_len: int = 10):
    return {
        "input_ids": torch.randint(1, 100, (batch_size, seq_len)),
        "labels": torch.cat(
            [
                torch.full((batch_size, seq_len // 2), -100),
                torch.randint(1, 100, (batch_size, seq_len // 2)),
            ],
            dim=1,
        ),
    }


def test_sft_sample_reads_multiple_batches():
    """Should read across batch boundaries to reach n, not stop after the first batch."""
    batches = [_make_sft_batch(2) for _ in range(5)]
    loader = iter(batches)
    samples = sample_from_dataloader(loader, n=5)
    assert len(samples) == 5
    assert all("input_ids" in s for s in samples)
    assert all("labels" in s for s in samples)


def test_sft_sample_respects_n():
    batches = [_make_sft_batch(10)]
    loader = iter(batches)
    samples = sample_from_dataloader(loader, n=3)
    assert len(samples) == 3


def test_sft_sample_fewer_than_n_available():
    batches = [_make_sft_batch(2)]
    loader = iter(batches)
    samples = sample_from_dataloader(loader, n=100)
    assert len(samples) == 2  # only 2 available


# ---------------------------------------------------------------------------
# sample_from_dataloader — DPO format
# ---------------------------------------------------------------------------


def _make_dpo_batch(num_examples: int, seq_len: int = 10):
    """DPO batches double the examples (chosen + rejected) and have completion_mask."""
    bs = num_examples * 2
    input_ids = torch.randint(1, 100, (bs, seq_len))
    completion_mask = torch.zeros(bs, seq_len, dtype=torch.long)
    completion_mask[:, seq_len // 2 :] = 1  # second half is completion
    return {"input_ids": input_ids, "completion_mask": completion_mask}


def test_dpo_batch_sampled_correctly():
    batches = [_make_dpo_batch(3)]
    loader = iter(batches)
    samples = sample_from_dataloader(loader, n=4)
    assert len(samples) == 3  # only 3 unique prompts from chosen side (6 rows // 2)
    assert all("completion_mask" in s for s in samples)
    assert all(s["labels"] is None for s in samples)


def test_dpo_samples_only_chosen_side():
    """DPO batch has 4 rows (2 chosen + 2 rejected). Should sample only 2 unique prompts."""
    batch = _make_dpo_batch(2)  # 2 examples → 4 rows
    loader = iter([batch])
    samples = sample_from_dataloader(loader, n=10)
    assert len(samples) == 2  # only 2 unique prompts from chosen side


def test_dpo_completion_mask_forwarded():
    batch = _make_dpo_batch(2)
    loader = iter([batch])
    samples = sample_from_dataloader(loader, n=2)
    assert samples[0]["completion_mask"] is not None
    assert samples[0]["completion_mask"].tolist() == batch["completion_mask"][0].tolist()


# ---------------------------------------------------------------------------
# sample_from_dataloader — GRPO format
# ---------------------------------------------------------------------------


def _make_grpo_batch(prompts: list[str]):
    return {"prompt": prompts}


def test_grpo_string_prompts_sampled():
    batches = [_make_grpo_batch(["Hello world", "How are you?", "Tell me a joke"])]
    loader = iter(batches)
    samples = sample_from_dataloader(loader, n=2)
    assert len(samples) == 2
    assert all("prompt_text" in s for s in samples)
    assert samples[0]["prompt_text"] == "Hello world"
    assert samples[1]["prompt_text"] == "How are you?"


def test_grpo_reads_multiple_batches():
    batches = [_make_grpo_batch(["p1", "p2"]), _make_grpo_batch(["p3", "p4"])]
    loader = iter(batches)
    samples = sample_from_dataloader(loader, n=3)
    assert len(samples) == 3
    assert [s["prompt_text"] for s in samples] == ["p1", "p2", "p3"]


def test_grpo_single_string_normalised():
    """A bare string in batch["prompt"] should still be sampled."""
    loader = iter([{"prompt": "a single prompt string"}])
    samples = sample_from_dataloader(loader, n=1)
    assert len(samples) == 1
    assert samples[0]["prompt_text"] == "a single prompt string"


# ---------------------------------------------------------------------------
# sample_from_dataloader — unknown format
# ---------------------------------------------------------------------------


def test_unknown_batch_format_returns_empty(recwarn):
    loader = iter([{"logits": torch.zeros(2, 10)}])
    samples = sample_from_dataloader(loader, n=5)
    assert samples == []
    assert any("unrecognised" in str(w.message).lower() for w in recwarn.list)
