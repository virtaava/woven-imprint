"""Minimal PEFT LoRA trainer for chat JSONL. Run ONLY with ~/ComfyUI/.venv/bin/python."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HF_DIR = Path.home() / "models" / "qwen3-4b-hf"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )


def build_example_tokens(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """Tokenize a chat; labels are -100 everywhere except assistant message tokens."""
    full_text = _render(tokenizer, messages, False)
    ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = _render(tokenizer, messages[:i], True)
        through = _render(tokenizer, messages[: i + 1], False)
        start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        end = len(tokenizer(through, add_special_tokens=False)["input_ids"])
        for j in range(start, min(end, len(ids))):
            labels[j] = ids[j]
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--mix", type=Path, default=None, help="second corpus mixed in (e.g. persona) ")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    rows = _load_rows(args.corpus) + (_load_rows(args.mix) if args.mix else [])
    examples = [e for e in (build_example_tokens(tok, r["messages"], args.max_len) for r in rows) if e]
    print(f"examples: {len(examples)} (from {len(rows)} rows)")

    model = AutoModelForCausalLM.from_pretrained(str(HF_DIR), dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(examples) / args.batch)
    total = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        L = max(len(e["input_ids"]) for e in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, e in enumerate(batch):
            n = len(e["input_ids"])
            ids[i, :n] = torch.tensor(e["input_ids"]); lab[i, :n] = torch.tensor(e["labels"]); att[i, :n] = 1
        return ids.cuda(), lab.cuda(), att.cuda()

    model.train(); step = 0; t0 = time.time(); loss_val = float("nan")
    torch.cuda.reset_peak_memory_stats()
    done = False
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for b in range(0, len(examples), args.batch):
            ids, lab, att = collate(examples[b:b + args.batch])
            loss = model(input_ids=ids, labels=lab, attention_mask=att).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; loss_val = loss.item()
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total} epoch {epoch} loss {loss_val:.4f} lr {sched.get_last_lr()[0]:.2e}")
            if step >= total:
                done = True; break
        if done:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    log = {"steps": step, "final_loss": loss_val, "wall_s": round(time.time() - t0, 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "examples": len(examples)}
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("saved", args.out, log)


if __name__ == "__main__":
    main()
