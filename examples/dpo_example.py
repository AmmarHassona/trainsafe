"""DPO training example with trainsafe and custom probes."""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from trainsafe import TrainSafeCallback

model_id = "Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id)
ref_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:500]")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(
        output_dir="./output_dpo",
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
            probes="probes_example.yaml",
            warn_threshold=0.7,
            stop_threshold=0.4,
            num_inference_samples=5,
            log_to_wandb=False,
        )
    ],
)

trainer.train()
