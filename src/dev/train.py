import argparse
import json
import random
import time
from functools import partial
from pathlib import Path
import contextlib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from safetensors.torch import save_file
from .data import SCHEMA_VERSION, read_rows, encode, collate
from .model import ThreeHeadEncoder, multitask_loss


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def move(batch, device):
    return {key: ({k: v.to(device) for k, v in value.items()} if isinstance(value, dict) else value.to(device))
            for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums, count = {}, {}
    def add(key, values):
        sums[key] = sums.get(key, 0.) + values.sum().item()
        count[key] = count.get(key, 0) + values.numel()
    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if (device.type == "cuda" and torch.cuda.is_bf16_supported())
                    else torch.autocast("mps", dtype=torch.float16) if device.type == "mps"
                    else contextlib.nullcontext())
    for batch in loader:
        batch = move(batch, device)
        with autocast_ctx:
            result = model(**batch)
        for key in ("choice", "score", "noul"):
            valid = batch["labels"][key] != -100
            if not valid.any():
                continue
            logits, labels = result[key][valid], batch["labels"][key][valid]
            if key == "noul":
                prob = logits.sigmoid()
                add("noul_brier", (prob-labels).square())
                add("noul_accuracy", ((prob >= .5) == labels.bool()).float())
            else:
                prob = logits.softmax(-1)
                add(key + "_accuracy", (prob.argmax(-1) == labels).float())
                truth = F.one_hot(labels.long(), prob.shape[-1]).float()
                add(key + "_brier", (prob - truth).square().sum(-1))
                clamped_prob = prob.gather(1, labels[:, None]).squeeze(1).float().clamp_min(1e-7)
                add(key + "_nll", -clamped_prob.log())
                if key == "score":
                    value = (prob * torch.arange(prob.shape[-1], device=device)).sum(-1)
                    add("score_mae", (value-labels).abs())
                    cdf_pred = prob.cumsum(-1)
                    cdf_truth = truth.cumsum(-1)
                    add("score_rps", (cdf_pred - cdf_truth).square().mean(-1))
                if key == "choice":
                    top = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
                    add("choice_recall_at_3", (top == labels[:, None]).any(-1).float())
    return {key: sums[key]/count[key] for key in sums}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--validation", default="data/validation.jsonl")
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--revision", help="Pinned Hugging Face model revision")
    p.add_argument("--output", default="runs/bootstrap")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=0, help="Optimizer steps; 0 means all epochs")
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--resume", help="Resume an epoch-boundary trainer.pt checkpoint")
    p.add_argument("--init-checkpoint", help="Warm-start model weights from an existing checkpoint directory")
    p.add_argument("--loss-mode", default="ce", choices=["ce", "brier"],
                   help="Loss objective: 'ce' for cross-entropy, 'brier' for proper quadratic calibration")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.max_length) <= 0 or args.max_steps < 0:
        p.error("epochs, batch size, accumulation, max length must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_for(args.device)
    train, validation = read_rows(args.train), read_rows(args.validation)
    if any(r.get('label', -100) == -100 for r in train + validation):
        raise ValueError('Training/evaluation require labels; annotation queues cannot be used as supervision')
    if {r['group'] for r in train} & {r['group'] for r in validation}:
        raise ValueError("Train/validation source groups overlap")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ThreeHeadEncoder.pretrained(args.model, revision=args.revision).to(device)
    if args.init_checkpoint:
        from safetensors.torch import load_file
        chk_p = Path(args.init_checkpoint) / "model.safetensors"
        if chk_p.exists():
            model.load_state_dict(load_file(chk_p), strict=False)
            print(f"Loaded warm-start weights from {chk_p}", flush=True)
        else:
            raise FileNotFoundError(f"Init checkpoint not found: {chk_p}")
    model.encoder.config.use_cache = False
    if args.lora_rank:
        if args.freeze_encoder:
            raise ValueError("Choose LoRA or frozen encoder, not both")
        if model.encoder.config.model_type not in {"qwen2", "qwen3"}:
            raise ValueError("LoRA targets in this prototype are configured for Qwen only")
        from peft import LoraConfig, get_peft_model
        model.encoder = get_peft_model(model.encoder, LoraConfig(
            r=args.lora_rank, lora_alpha=2*args.lora_rank, lora_dropout=.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        if args.gradient_checkpointing:
            model.encoder.enable_input_require_grads()
    if args.max_length > model.encoder.config.max_position_embeddings:
        raise ValueError("max-length exceeds encoder context")
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    elif args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    train = [encode(r, tokenizer, args.max_length) for r in train]
    validation = [encode(r, tokenizer, args.max_length) for r in validation]
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW([
        {"params": [v for n, v in model.named_parameters() if n.startswith("encoder.") and v.requires_grad], "lr": args.encoder_lr},
        {"params": [v for n, v in model.named_parameters() if not n.startswith("encoder.")], "lr": args.head_lr},
    ], weight_decay=.01)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    start_epoch, steps = 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get('schema_version') != SCHEMA_VERSION or saved.get('revision') != args.revision:
            raise ValueError('Resume schema version and model revision must match')
        if saved["model_name"] != args.model or saved["freeze_encoder"] != args.freeze_encoder:
            raise ValueError("Resume model and freeze policy must match")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        start_epoch, steps = saved["next_epoch"], saved["steps"]
        torch.set_rng_state(saved["rng"])
    started = time.perf_counter()
    status = "calibrated" if args.loss_mode == "brier" else "experimental_uncalibrated"
    metadata = {**vars(args), 'schema_version': SCHEMA_VERSION, "device_used": str(device), "parameters": sum(x.numel() for x in model.parameters()),
                "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
                "status": status, "before": evaluate(model, val_loader, device)}
    print(json.dumps(metadata), flush=True)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if (device.type == "cuda" and torch.cuda.is_bf16_supported())
                        else torch.autocast("mps", dtype=torch.float16) if device.type == "mps"
                        else contextlib.nullcontext())
        for i, batch in enumerate(train_loader):
            batch = move(batch, device)
            # Normalize the last partial accumulation window correctly.
            window = min(args.accumulation, len(train_loader) - (i // args.accumulation)*args.accumulation)
            with autocast_ctx:
                loss = multitask_loss(model(**batch), batch["labels"], loss_mode=args.loss_mode)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            (loss/window).backward()
            if (i+1) % args.accumulation == 0 or i+1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
                if device.type == "mps" and steps % 10 == 0:
                    torch.mps.empty_cache()
                print(json.dumps({"epoch": epoch+1, "step": steps, "loss": round(loss.item(), 4),
                                  "elapsed_s": round(time.perf_counter()-started, 1)}), flush=True)
                if steps % 200 == 0:
                    tokenizer.save_pretrained(out / "tokenizer")
                    model.encoder.config.save_pretrained(out / "encoder")
                    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors")
                    (out / "run.json").write_text(json.dumps({**metadata, "steps": steps, "elapsed_s": time.perf_counter()-started}, indent=2))
                if args.max_steps and steps >= args.max_steps:
                    stop = True; break
        metrics = evaluate(model, val_loader, device)
        metadata.update({"steps": steps, "after": metrics, "elapsed_s": time.perf_counter()-started})
        tokenizer.save_pretrained(out / "tokenizer")
        model.encoder.config.save_pretrained(out / "encoder")
        if args.lora_rank:
            model.encoder.save_pretrained(out / "adapter")
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
                       if not k.startswith("encoder.")}, out / "heads.safetensors")
        else:
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors")
        (out / "run.json").write_text(json.dumps(metadata, indent=2))
        # Interrupted mid-epoch exports inference weights, not a misleading resumable checkpoint.
        if i+1 == len(train_loader):
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": steps,
                        'schema_version': SCHEMA_VERSION, 'revision': args.revision,
                        "model_name": args.model, "freeze_encoder": args.freeze_encoder,
                        "next_epoch": epoch+1, "rng": torch.get_rng_state()}, out / "trainer.pt")
        print(json.dumps({"validation": metrics}), flush=True)
        if stop:
            break


if __name__ == "__main__":
    main()
