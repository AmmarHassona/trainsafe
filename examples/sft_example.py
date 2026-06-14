"""Minimal SFT training example with trainsafe.

First run will download Qwen2-0.5B-Instruct (~1GB). Swap in any other
causal LM — the callback works regardless of architecture.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from trainsafe import TrainSafeCallback

model_id = "Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("trl-lib/Capybara", split="train[:500]")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./output",
        eval_strategy="steps",
        eval_steps=100,
        max_steps=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        report_to="none",
    ),
    train_dataset=dataset,
    eval_dataset=dataset.select(range(50)),
    callbacks=[
        TrainSafeCallback(
            warn_threshold=0.7,
            stop_threshold=0.4,
            num_inference_samples=5,
            log_to_wandb=False,
        )
    ],
)

trainer.train()
