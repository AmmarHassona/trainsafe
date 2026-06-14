from __future__ import annotations

import warnings

import torch


def extract_prompt_ids(
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    completion_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return only the prompt portion of input_ids.

    Supports three token-masking conventions used by different TRL trainers:

    - SFT (``labels``): prompt tokens are masked to -100; slice up to the
      first non-(-100) position.
    - DPO (``completion_mask``): 0 = prompt token, 1 = completion token;
      slice up to the first 1.
    - Neither: treat the entire sequence as the prompt (e.g. GRPO batches
      where prompts are handled separately as raw strings).
    """
    if completion_mask is not None:
        for i, m in enumerate(completion_mask):
            if m.item() == 1:
                return input_ids[:i]
        # All zeros — no completion tokens found; return full sequence.
        return input_ids

    if labels is not None:
        for i, label in enumerate(labels):
            if label.item() != -100:
                # i == 0 means no prompt prefix at all — return empty rather
                # than the full sequence (the original bug).
                return input_ids[:i]
        # All -100 — entire sequence is the prompt.
        return input_ids

    return input_ids


def generate_output(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 256,
) -> str:
    """Run greedy generation from prompt_ids; return decoded new tokens only."""
    # Truncate prompt if it would leave no room for new tokens.
    max_model_len = getattr(tokenizer, "model_max_length", None)
    if max_model_len and max_model_len < 1e6:  # ignore sentinel values like 1e30
        max_prompt = max_model_len - max_new_tokens
        if max_prompt > 0 and len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]  # keep the most recent tokens

    # model.device works for both single-device and device_map models
    # (PreTrainedModel.device returns the device of the first parameter).
    device = model.device
    ids = prompt_ids.unsqueeze(0).to(device)
    mask = torch.ones_like(ids)
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

    new_tokens = out[0, ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def sample_from_dataloader(dataloader, n: int) -> list[dict]:
    """Sample up to *n* examples from the eval dataloader.

    Handles the three batch formats emitted by HuggingFace / TRL trainers:

    - **SFT** (``transformers.Trainer`` / ``trl.SFTTrainer``):
      batch has ``"input_ids"`` + ``"labels"`` with -100 prompt masking.
    - **DPO** (``trl.DPOTrainer``):
      batch has ``"input_ids"`` + ``"completion_mask"`` (0=prompt, 1=completion).
      The batch size is 2× the number of examples (chosen + rejected rows).
      No ``"labels"`` key.
    - **GRPO** (``trl.GRPOTrainer``):
      batch has a raw ``"prompt"`` string column (or list-of-dicts for
      conversational format); no ``"input_ids"`` at this stage.

    Reads as many batches as needed to collect *n* samples rather than
    stopping after the first batch (which would silently under-collect when
    ``per_device_eval_batch_size < n``).
    """
    samples: list[dict] = []

    for batch in dataloader:
        # GRPO (TRL 1.x): eval dataloader yields a list of dicts,
        # e.g. [{"prompt": "..."}, {"prompt": "..."}]
        if isinstance(batch, (list, tuple)):
            for item in batch:
                if isinstance(item, dict) and "prompt" in item:
                    samples.append({"prompt_text": item["prompt"]})
                elif isinstance(item, str):
                    samples.append({"prompt_text": item})
                if len(samples) >= n:
                    return samples

        elif "completion_mask" in batch and "input_ids" in batch:
            # DPO: batch = [chosen_0,..., rejected_0,...] — sample chosen side only
            total_rows = batch["input_ids"].shape[0]
            unique_rows = total_rows // 2
            for i in range(unique_rows):
                samples.append({
                    "input_ids": batch["input_ids"][i].cpu(),
                    "labels": None,
                    "completion_mask": batch["completion_mask"][i].cpu(),
                })
                if len(samples) >= n:
                    return samples

        elif "input_ids" in batch:
            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size):
                samples.append({
                    "input_ids": batch["input_ids"][i].cpu(),
                    "labels": batch["labels"][i].cpu() if "labels" in batch else None,
                    "completion_mask": None,
                })
                if len(samples) >= n:
                    return samples

        elif "prompt" in batch:
            # GRPO-style dict batch: {"prompt": ["...", "..."]}
            raw_prompts = batch["prompt"]
            if isinstance(raw_prompts, str):
                raw_prompts = [raw_prompts]
            for prompt in raw_prompts:
                samples.append({"prompt_text": prompt})
                if len(samples) >= n:
                    return samples

        else:
            fmt = list(batch.keys()) if hasattr(batch, "keys") else type(batch).__name__
            warnings.warn(
                f"TrainSafe: unrecognised eval batch format ({fmt}) — skipping sample collection."
            )
            break

        if len(samples) >= n:
            break

    return samples
