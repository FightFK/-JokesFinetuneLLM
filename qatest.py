#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jokes ChatML Streaming Tester (Qwen3-1.7B + LoRA)
- Loads base model + optional LoRA adapter (or a merged dir)
- Uses tokenizer.apply_chat_template() for ChatML
- Streams assistant tokens to terminal
- Jokes-aware prompt builder:
    Mode: riddles | wordplay | captions
    - riddles:  user ใส่ "คำถามมุกทายปัญหา" -> assistant ตอบ "คำเฉลยสั้น ๆ"
    - wordplay: user ใส่ "หัวข้อ/ธีม" -> assistant สร้างมุกเล่นคำสั้น ๆ
    - captions: user ใส่ "บริบท/รูป/อารมณ์" -> assistant สร้างแคปชั่นฮา ๆ
"""

import os
import sys
import re
import threading
import argparse
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
)
from peft import PeftModel

# -----------------------------
# CLI
# -----------------------------
def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B",
                    help="Base model name/path (HF hub or local).")
    ap.add_argument("--adapter", default="./qwen3-1p7b-jokes-lora",
                    help="LoRA adapter directory (optional).")
    ap.add_argument("--merged", action="store_true",
                    help="If provided, skip LoRA loading (assume merged).")
    ap.add_argument("--model_dir", default="",
                    help="If you exported a fully-merged model dir, set it here (overrides --base/--adapter).")
    ap.add_argument("--system", default="You are a funny assistant. Answer with jokes in a Q&A style.",
                    help="System prompt.")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    return ap.parse_args()

args = build_args()

# -----------------------------
# Device
# -----------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32
print(f"[INFO] Device: {DEVICE} | DType: {DTYPE}")

# -----------------------------
# Load tokenizer
# -----------------------------
def load_tokenizer():
    tk_path_candidates = [p for p in [args.model_dir, args.adapter, args.base] if p]
    last_err = None
    for p in tk_path_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(p, use_fast=True, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load tokenizer. Last error: {last_err}")

tokenizer = load_tokenizer()

# -----------------------------
# Load model (+LoRA if needed)
# -----------------------------
def load_model():
    if args.model_dir:
        print(f"[INFO] Loading merged model from: {args.model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=DTYPE, trust_remote_code=True
        )
        return model

    print(f"[INFO] Loading base model: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=DTYPE, trust_remote_code=True
    )

    if not args.merged and os.path.isdir(args.adapter):
        try:
            print(f"[INFO] Loading LoRA adapter: {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)
        except Exception as e:
            print(f"[WARN] Could not load LoRA adapter ({e}). Proceeding with base only.")
    else:
        print("[INFO] Skipping LoRA load (merged mode or adapter dir not set).")

    return model

model = load_model()
model.config.use_cache = True
if DEVICE != "cpu":
    model.to(DEVICE)
model.eval()

# end tokens (Qwen may use <|im_end|>)
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tokenizer.get_vocab() else None
EOS_IDS = [tokenizer.eos_token_id] + ([im_end_id] if im_end_id and im_end_id != tokenizer.eos_token_id else [])

print("\n✅ Jokes model ready! Type 'exit' to quit.\n")

# -----------------------------
# Prompt builder (Jokes)
# -----------------------------
JOKES_INSTRUCTION = (
    "ตอบเป็นภาษาไทย, สั้น, กระชับ, และเล่นคำให้ฮา.\n"
    "- riddles: ตอบเฉลยสั้น ๆ เท่านั้น, ไม่อธิบายยาว.\n"
    "- wordplay: ยิงมุก 1-2 ประโยค.\n"
    "- captions: ให้แคปชั่น 1 ประโยค."
)

def build_messages(user_input: str, mode: str = "riddles"):
    mode = mode.lower().strip()
    if mode not in {"riddles", "wordplay", "captions"}:
        mode = "riddles"

    user_block = f"[Jokes]\nMode: {mode}\nInstruction: {JOKES_INSTRUCTION}\nInput: {user_input}"
    return [
        {"role": "system", "content": args.system},
        {"role": "user", "content": user_block}
    ]

# -----------------------------
# Chat Once
# -----------------------------
def chat_once(mode: str, user_input: str):
    messages = build_messages(user_input, mode)
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if DEVICE != "cpu":
        input_ids = input_ids.to(DEVICE)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        do_sample=False,          # greedy
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        streamer=streamer,
        eos_token_id=EOS_IDS if len(EOS_IDS) > 1 else EOS_IDS[0],
        pad_token_id=tokenizer.eos_token_id,
    )

    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    output_text = ""
    for piece in streamer:
        output_text += piece
    t.join()

    # 🔑 Clean reasoning tags <think> ... </think>
    output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

    print(f"A: {output_text}")

# -----------------------------
# Interactive loop
# -----------------------------
def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        return "exit"

while True:
    q = ask("Input (riddle/caption/wordplay content, or 'exit'): ").strip()
    if q.lower() in {"exit", "quit"}:
        break
    if not q:
        continue

    m = ask("Mode [riddles|wordplay|captions] (default riddles): ").strip().lower() or "riddles"
    try:
        chat_once(m, q)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
