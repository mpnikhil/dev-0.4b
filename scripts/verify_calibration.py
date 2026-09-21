"""Verify temperature-scaling calibration for Dev (dev-0.4b).

Evaluates a single checkpoint's three readouts (noul, choice, score) at T=1 (raw SFT)
vs the fitted per-readout temperatures, on held-out data. Unlike the fitter, this also
reports:
  * the ordinal SCORE readout's expected-value MAE, which temperature scaling shifts
    (mu = sum_j j * p_j changes when the softmax is sharpened/flattened);
  * ECE with BOTH fixed-width and equal-mass (adaptive) bins, so a single dominant
    confidence bin cannot hide (or manufacture) a calibration effect;
  * a reliability diagram per readout.

Temperatures are read from the checkpoint's run.json ("temperatures") unless overridden
on the CLI, so this works before/without --apply-to-checkpoint has been run.

Kept intentionally self-contained (its own logit collector) to stay robust while
scripts/calibrate_temperature.py is edited concurrently; the two collectors share the
model.forward contract in src/dev/model.py, nothing else.
"""
import argparse
import contextlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from safetensors.torch import load_file

from dev.model import ThreeHeadEncoder
from dev.data import read_rows, encode, collate
from dev.train import device_for, move


# --------------------------------------------------------------------------- #
# Logit collection (one forward pass over the validation set; T applied later) #
# --------------------------------------------------------------------------- #
def collect_logits(checkpoint_dir, val_file, device="mps", batch_size=16, limit=None, fp32=True):
    path = Path(checkpoint_dir)
    dev = device_for(device)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    config = AutoConfig.from_pretrained(path / "encoder", local_files_only=True)
    model = ThreeHeadEncoder(AutoModel.from_config(config, attn_implementation="sdpa"))
    model.load_state_dict(load_file(path / "model.safetensors"), strict=False)
    model.to(dev).eval()

    rows = read_rows(val_file)
    if limit:
        rows = rows[:limit]
    encoded = [encode(r, tokenizer, 1024) for r in rows]
    loader = DataLoader(encoded, batch_size=batch_size,
                        collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))

    data = {k: {"logits": [], "labels": []} for k in ("noul", "choice", "score")}
    # fp32 for the calibration logit dump: accuracy > speed on a small val set, and we
    # do not want autocast rounding perturbing the very logits we are calibrating.
    autocast_ctx = (contextlib.nullcontext() if fp32 or dev.type != "mps"
                    else torch.autocast("mps", dtype=torch.float16))

    with torch.no_grad():
        for batch in loader:
            batch = move(batch, dev)
            with autocast_ctx:
                result = model(**batch)

            noul_valid = batch["labels"]["noul"] != -100
            if noul_valid.any():
                data["noul"]["logits"].extend(result["noul"][noul_valid].cpu().float().tolist())
                data["noul"]["labels"].extend(batch["labels"]["noul"][noul_valid].cpu().tolist())

            for kind in ("choice", "score"):
                valid = batch["labels"][kind] != -100
                if not valid.any():
                    continue
                logits = result[kind][valid].cpu().float()
                labels = batch["labels"][kind][valid].cpu().tolist()
                cand_valid = result["candidate_valid"][valid].cpu()
                for i in range(len(labels)):
                    n_cands = int(cand_valid[i].sum().item())
                    data[kind]["logits"].append(logits[i, :n_cands].tolist())
                    data[kind]["labels"].append(int(labels[i]))
    return data


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _ece(conf, correct, n_bins, adaptive):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return 0.0, []
    bins = []
    ece = 0.0
    if adaptive:  # equal-mass: each bin holds ~n/n_bins samples
        order = np.argsort(conf)
        conf, correct = conf[order], correct[order]
        edges = np.linspace(0, n, n_bins + 1).astype(int)
        spans = [(edges[b], edges[b + 1]) for b in range(n_bins)]
    else:         # fixed-width over [min possible conf, 1.0]
        lo = float(conf.min())
        bounds = np.linspace(min(lo, 0.5), 1.0, n_bins + 1)
        spans = []
        for b in range(n_bins):
            last = b == n_bins - 1
            m = (conf >= bounds[b]) & (conf <= bounds[b + 1] if last else conf < bounds[b + 1])
            spans.append(np.where(m)[0])
    for span in spans:
        idx = np.arange(span[0], span[1]) if adaptive else span
        if len(idx) == 0:
            continue
        c, y = conf[idx].mean(), correct[idx].mean()
        gap = abs(y - c)
        ece += len(idx) / n * gap
        bins.append({"count": int(len(idx)), "conf": float(c), "acc": float(y), "gap": float(gap)})
    return float(ece), bins


def metrics(kind, logits, labels, T, n_bins=10):
    labels = np.asarray(labels)
    out = {"temperature": float(T), "n": int(len(labels))}
    if kind == "noul":
        p = 1.0 / (1.0 + np.exp(-np.asarray(logits, float) / T))
        conf = np.maximum(p, 1.0 - p)
        correct = ((p >= 0.5).astype(float) == labels).astype(float)
        out["brier"] = float(np.mean((p - labels) ** 2))
        pc = np.clip(np.where(labels == 1, p, 1 - p), 1e-12, 1.0)
        out["nll"] = float(-np.mean(np.log(pc)))
    else:
        probs = [np.exp((np.asarray(lg, float) - np.max(lg)) / T) /
                 np.sum(np.exp((np.asarray(lg, float) - np.max(lg)) / T)) for lg in logits]
        conf = np.array([p.max() for p in probs])
        preds = np.array([int(p.argmax()) for p in probs])
        correct = (preds == labels).astype(float)
        out["brier"] = float(np.mean([np.sum((p - np.eye(len(p))[l]) ** 2)
                                      for p, l in zip(probs, labels)]))
        out["nll"] = float(-np.mean([np.log(np.clip(p[l], 1e-12, 1.0))
                                     for p, l in zip(probs, labels)]))
        if kind == "score":  # ordinal expected value (0-indexed, matches inference.py)
            mu = np.array([float(np.dot(np.arange(len(p)), p)) for p in probs])
            out["mae"] = float(np.mean(np.abs(mu - labels)))
            out["exact_acc"] = float(np.mean(np.round(mu) == labels))
    out["accuracy"] = float(np.mean(correct))
    out["mean_confidence"] = float(np.mean(conf))
    out["ece_fixed"], _ = _ece(conf, correct, n_bins, adaptive=False)
    out["ece_adaptive"], out["reliability"] = _ece(conf, correct, n_bins, adaptive=True)
    return out


def _fmt(kind, u, c):
    lines = [f"\n[{kind.upper()}]  ({u['n']} samples, T={c['temperature']:.4f})"]
    row = lambda name, a, b, lower: (
        f"  {name:<20} | {a:>10.4f} -> {b:>10.4f} | "
        f"{'delta ' + format(b - a, '+.4f'):<14} {'[WIN]' if lower and b < a else ''}")
    lines.append(row("Accuracy", u["accuracy"], c["accuracy"], False)
                 + ("  [INVARIANT]" if abs(u["accuracy"] - c["accuracy"]) < 1e-9 else "  [!! moved]"))
    lines.append(row("NLL", u["nll"], c["nll"], True))
    lines.append(row("Brier", u["brier"], c["brier"], True))
    lines.append(row("ECE (fixed bins)", u["ece_fixed"], c["ece_fixed"], True))
    lines.append(row("ECE (equal-mass)", u["ece_adaptive"], c["ece_adaptive"], True))
    lines.append(f"  {'Mean confidence':<20} | {u['mean_confidence']:>10.4f} -> {c['mean_confidence']:>10.4f}")
    if kind == "score":
        lines.append(row("Expected-value MAE", u["mae"], c["mae"], True)
                     + "  <- temperature shifts E[j]; must not regress")
        lines.append(row("Exact accuracy", u["exact_acc"], c["exact_acc"], False))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/dev-0.4b")
    parser.add_argument("--val-data", default="data/validation_refine.jsonl")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--t-noul", type=float, default=None)
    parser.add_argument("--t-choice", type=float, default=None)
    parser.add_argument("--t-score", type=float, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    meta = json.loads((Path(args.checkpoint) / "run.json").read_text())
    fitted = meta.get("temperatures", {}) or {}
    temps = {
        "noul": args.t_noul if args.t_noul is not None else fitted.get("noul", 1.0),
        "choice": args.t_choice if args.t_choice is not None else fitted.get("choice", 1.0),
        "score": args.t_score if args.t_score is not None else fitted.get("score", 1.0),
    }

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperatures (T=1.0 means uncalibrated / not yet fitted): {json.dumps(temps)}")
    data = collect_logits(args.checkpoint, args.val_data, device=args.device, limit=args.limit)

    print("\n" + "=" * 82)
    print("        CALIBRATION VERIFICATION  (raw SFT  ->  temperature-scaled readout)")
    print("=" * 82)

    report = {"checkpoint": args.checkpoint, "temperatures": temps, "readouts": {}}
    for kind in ("noul", "choice", "score"):
        if not data[kind]["labels"]:
            continue
        uncal = metrics(kind, data[kind]["logits"], data[kind]["labels"], 1.0, args.n_bins)
        cal = metrics(kind, data[kind]["logits"], data[kind]["labels"], temps[kind], args.n_bins)
        report["readouts"][kind] = {"uncalibrated": uncal, "calibrated": cal}
        print(_fmt(kind, uncal, cal))

    print("\n" + "=" * 82)
    print("RELIABILITY (equal-mass bins) per readout: conf -> acc (gap), n")
    for kind, blk in report["readouts"].items():
        print(f"\n  {kind.upper()} @ T={temps[kind]:.3f}")
        for b in blk["calibrated"]["reliability"]:
            print(f"    conf {b['conf']:.3f} -> acc {b['acc']:.3f}  (gap {b['gap']:.3f}, n={b['count']})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
