import os
from typing import cast

import torch
from datasets import Dataset, load_dataset
from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOConfig, ORPOTrainer

device = torch.device("mps") if torch.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    os.environ["BASE_MODEL"], torch_dtype=torch.bfloat16
).to(device)

tokenizer = AutoTokenizer.from_pretrained(os.environ["BASE_MODEL"])
train_dataset: Dataset = cast(
    Dataset, load_dataset(os.environ["DATASET"], split="train")
)


peft_config = LoraConfig(
    r=8,  # Rank of LoRA matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config).to(device)
peft_model.print_trainable_parameters()


training_args = ORPOConfig(
    optim="adamw_torch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    bf16=True,
    output_dir=f"trained/{os.environ['BASE_MODEL']}-ORPO",
    logging_steps=10,
    report_to="mlflow",
)

trainer = ORPOTrainer(
    model=peft_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)


if __name__ == "__main__":
    trainer.train()
