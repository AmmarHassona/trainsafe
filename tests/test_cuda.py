"""CUDA smoke tests — run on a machine with a CUDA GPU to verify GPU behavior.

Skip automatically when CUDA is not available.

Usage (Colab / RunPod / any CUDA machine):
    pip install "trainsafe[trl]" datasets
    pytest tests/test_cuda.py -v
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

pytest.importorskip("trl", reason="trl not installed")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from trainsafe import TrainSafeCallback

MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def cuda_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def test_generate_output_on_cuda(cuda_model_and_tokenizer):
    """generate_output moves tensors to GPU and returns a string."""
    from trainsafe.utils import generate_output

    model, tokenizer = cuda_model_and_tokenizer
    prompt_ids = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"][0]
    output = generate_output(model, tokenizer, prompt_ids, max_new_tokens=32)
    assert isinstance(output, str)
    assert len(output) > 0


def test_cuda_cache_cleared_after_inference(cuda_model_and_tokenizer):
    """Memory used before and after a callback run should be similar (cache cleared)."""
    from trainsafe.utils import generate_output

    model, tokenizer = cuda_model_and_tokenizer
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()

    prompt_ids = tokenizer("Test prompt", return_tensors="pt")["input_ids"][0]
    generate_output(model, tokenizer, prompt_ids, max_new_tokens=64)

    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    # Allow 10MB headroom for any persistent allocations
    assert after - before < 10 * 1024 * 1024, f"Memory grew by {(after - before) / 1e6:.1f}MB after cache clear"


def test_sft_callback_on_cuda():
    """Full SFTTrainer run on GPU — no crash, checks fire."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset("trl-internal-testing/zen", "standard_prompt_completion")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="/tmp/trainsafe_cuda_sft",
            eval_strategy="steps",
            eval_steps=5,
            max_steps=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            report_to="none",
            fp16=True,
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        callbacks=[TrainSafeCallback(num_inference_samples=2, max_new_tokens=32, log_to_wandb=False)],
    )
    trainer.train()


def test_long_prompt_truncation_on_cuda(cuda_model_and_tokenizer):
    """Prompts longer than model_max_length - max_new_tokens are truncated, not errored."""
    from trainsafe.utils import generate_output

    model, tokenizer = cuda_model_and_tokenizer
    # Create a prompt that's intentionally too long
    tokenizer.model_max_length = 64  # temporarily shrink the limit
    long_prompt = "word " * 200
    prompt_ids = tokenizer(long_prompt, return_tensors="pt")["input_ids"][0]

    output = generate_output(model, tokenizer, prompt_ids, max_new_tokens=32)
    assert isinstance(output, str)
    tokenizer.model_max_length = 32768  # restore
