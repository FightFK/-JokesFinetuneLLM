#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA SFT for Qwen3-1.7B on Mac (MPS) using ChatML JSONL
- Expects JSONL with {"messages":[{role,content}, ...], ...}
- Uses tokenizer.apply_chat_template() to format prompts
- Masks loss on system/user tokens; trains only on assistant content
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, default_data_collator
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# Paths & hyperparams
# -----------------------------
MODEL_NAME   = "Qwen/Qwen3-1.7B"
DATA_PATH    = "jokes_dataset.jsonl"   
OUTPUT_DIR   = "./qwen3-1p7b-jokes-lora"

BLOCK_SIZE   = 1024
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LR           = 2e-4
EPOCHS       = 8        
WARMUP_RATIO = 0.1
LOG_STEPS    = 10
SAVE_STEPS   = 200

# -----------------------------
# Device & dtype
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# -----------------------------
# Tokenizer & model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
)
model.config.use_cache = False

# Enable LoRA (attention + MLP)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# -----------------------------
# Dataset
# -----------------------------
data = load_dataset("json", data_files=DATA_PATH, split="train")

ASSISTANT_ROLE = "assistant"

def build_example(example):
    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return {}

    # full conversation (system+user+assistant)
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # prompt-only (system+user, assistant="")
    prompt_only_messages = []
    for m in messages:
        if m.get("role") == ASSISTANT_ROLE:
            prompt_only_messages.append({"role": ASSISTANT_ROLE, "content": ""})
            break
        else:
            prompt_only_messages.append(m)
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    full = tokenizer(full_text, truncation=True, max_length=BLOCK_SIZE)
    prompt = tokenizer(prompt_text, truncation=True, max_length=BLOCK_SIZE)

    input_ids = full["input_ids"]
    attn_mask = full["attention_mask"]

    prefix_len = len(prompt["input_ids"])
    if prefix_len > len(input_ids):
        prefix_len = len(input_ids)

    labels = input_ids.copy()
    for i in range(prefix_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels
    }

tokenized = data.map(build_example, remove_columns=data.column_names)

# -----------------------------
# Trainer
# -----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=(device == "cuda"),
    bf16=False,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    num_train_epochs=EPOCHS,    
    warmup_ratio=WARMUP_RATIO,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=default_data_collator,
)

trainer.train()

# -----------------------------
# Save LoRA adapter + tokenizer
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ LoRA SFT finished. Adapter saved at:", OUTPUT_DIR)
print("👉 Load for inference:")
print("   from transformers import AutoModelForCausalLM, AutoTokenizer")
print("   from peft import PeftModel")
print(f"   tok = AutoTokenizer.from_pretrained('{MODEL_NAME}')")
print(f"   base = AutoModelForCausalLM.from_pretrained('{MODEL_NAME}')")
print(f"   model = PeftModel.from_pretrained(base, '{OUTPUT_DIR}')")
