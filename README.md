# trainsafe

[![PyPI](https://img.shields.io/pypi/v/trainsafe)](https://pypi.org/project/trainsafe/)
[![Python](https://img.shields.io/pypi/pyversions/trainsafe)](https://pypi.org/project/trainsafe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/AmmarHassona/trainsafe/actions/workflows/ci.yml/badge.svg)](https://github.com/AmmarHassona/trainsafe/actions/workflows/ci.yml)

Behavioral health checks for HuggingFace / TRL fine-tuning.

---

The idea behind this project occurred to me when fine-tuning a model on another languages, the loss looked fine the whole run, but when training finished, the model was outputting a completly different language.

Loss going down doesn't mean the model is behaving correctly. `trainsafe` hooks into your eval loop, generates a handful of outputs at each checkpoint, and checks whether they're still in the right language, format, and length. If something looks wrong, it warns you. If it looks bad enough, it stops training and points you at the last healthy checkpoint.

## Install

```bash
pip install trainsafe

# with W&B logging
pip install "trainsafe[wandb]"

# with language drift detection (adds langdetect)
pip install "trainsafe[language]"
```

## Usage

```python
from trainsafe import TrainSafeCallback

trainer = SFTTrainer(
    model=model,
    ...
    callbacks=[TrainSafeCallback()]
)
trainer.train()
```

Works with `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, and the base `Trainer`.

## What it checks

At each eval checkpoint, trainsafe generates a small sample of outputs (default: 5) and runs five checks automatically:

**Language** — detects if the model switches output language mid-training. Requires `trainsafe[language]`; silently skipped if not installed.

**Length** — catches output collapse (suddenly generating much shorter text) or runaway growth. Compares against a rolling baseline so legitimate learning doesn't trigger false alarms.

**Repetition** — flags n-gram loops inside individual outputs (the classic "the the the the" failure mode).

**Echo** — flags outputs that are mostly a copy of the prompt rather than a response.

**Format** — detects if a model trained to output JSON starts producing plain text, or vice versa. Also adapts when format changes consistently, so intentional format learning doesn't keep alarming.

Health score is the average of all check scores. Below `warn_threshold` (default 0.7), a warning is logged. Below `stop_threshold` (default 0.4), training stops.

## Configuration

```python
TrainSafeCallback(
    probes="probes.yaml",        # path to custom probe file, optional
    warn_threshold=0.7,
    stop_threshold=0.4,
    num_inference_samples=5,     # bump to 15-20 for more reliable signal
    max_new_tokens=256,          # tune to your task — Q&A tasks need far fewer
    probe_every_n_steps=None,    # defaults to every eval step
    log_to_wandb=True,
)
```

## Custom probes

Fixed prompt-level assertions in YAML, evaluated at every checkpoint:

```yaml
probes:
  - prompt: "مرحبا، كيف يمكنني مساعدتك؟"
    checks:
      - language: ar
      - min_length: 10
      - not_contains: ["<|im_start|>", "###"]

  - prompt: "اشرح لي ما هو التعلم الآلي"
    checks:
      - language: ar
      - coherent: true
```

Available checks: `language`, `min_length`, `max_length`, `contains`, `not_contains`, `format` (`json` / `markdown` / `plain`), `coherent` (flags empty, too-short, or heavily repetitive outputs).

Probes are particularly useful when you have a specific capability you can't afford to lose.

## Terminal output

SFT run (healthy model, `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, default settings):

```
[TrainSafe @ step 5] ✅ Language consistent (en)
[TrainSafe @ step 5] ✅ Output length normal (avg 62 words)
[TrainSafe @ step 5] ✅ No repetition detected
[TrainSafe @ step 5] ✅ No prompt echoing
[TrainSafe @ step 5] ✅ Format consistent (plain)
[TrainSafe @ step 5] Overall health: 1.00
```

DPO run (same model, `standard_preference` dataset) — same output, confirming DPO batch format is handled correctly.

When something goes wrong (language drift example):

```
[TrainSafe @ step 600] 🚨 Language drift — expected ar, got zh
[TrainSafe @ step 600] 🚨 Output length collapsed (avg 3 words vs baseline 87)
[TrainSafe @ step 600] ⚠️  Repetition detected in 3/5 outputs
[TrainSafe @ step 600] Overall health: 0.20
>>> TrainSafe stopped training. Recommended checkpoint: step 400.
```

## Compute overhead

trainsafe runs `model.generate()` on `num_inference_samples` prompts after each eval. This is pure inference — no gradients, no weight updates, CUDA cache is cleared after each run.

The cost scales with model size and `max_new_tokens` (GPU estimates):

| Model size | max_new_tokens | overhead per checkpoint |
|---|---|---|
| <1B | 256 (default) | <5s |
| 7B | 256 | ~10–20s |
| 7B | 64 | ~3–5s |
| 70B | 256 | minutes |

For large models, set `max_new_tokens` to match your actual task length (e.g. 32 for short-answer tasks) and use `probe_every_n_steps` to check less often than you evaluate.

## Limitations

Tested on CPU and single NVIDIA GPU (T4). **Distributed training (DeepSpeed, FSDP, multi-GPU via `device_map="auto"`)** is untested and may not work correctly, the callback receives a wrapped model in those cases and `model.device` may not behave as expected.

## W&B metrics

When a W&B run is active, trainsafe logs `trainsafe/language_consistency`, `trainsafe/avg_output_length`, `trainsafe/repetition_rate`, `trainsafe/echo_rate`, `trainsafe/format_consistency`, `trainsafe/custom_probe_pass_rate` (if probes are configured), and `trainsafe/overall_health`.
