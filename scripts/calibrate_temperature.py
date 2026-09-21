"""Temperature scaling calibration for Dev (dev-0.4b).

Fits 3 task-specific temperatures (T_noul, T_choice, T_score) on held-out validation
data by minimizing Negative Log-Likelihood (NLL) via PyTorch (Guo et al. 2017).

Since Dev uses a single universal choice head, logit scales naturally differ across:
1. Binary contrast (Noul: s_yes - s_no)
2. Multi-class categorical (Choice: 77-way Banking, etc.)
3. Ordinal levels (Score: 5-star ratings)

Fitting three independent scalar readout temperatures preserves 100% of the argmax
predictions and rankings while properly softening overconfident probabilities.
"""

import contextlib
import json
from pathlib import Path
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from safetensors.torch import load_file
from dev.model import ThreeHeadEncoder
from dev.data import SCHEMA_VERSION, read_rows, encode, collate
from dev.train import device_for, move


def compute_ece_and_brier(probs, labels, n_bins=10, is_binary=False):
    """Compute ECE, Brier Score, and mean confidence."""
    if is_binary:
        # Binary: probs is p_yes in [0, 1], labels in {0, 1}
        conf = np.maximum(probs, 1.0 - probs)
        preds = (probs >= 0.5).astype(float)
        correct = (preds == labels).astype(float)
        brier = np.mean((probs - labels) ** 2)
        nll = -np.mean(labels * np.log(np.clip(probs, 1e-12, 1.0)) + (1 - labels) * np.log(np.clip(1 - probs, 1e-12, 1.0)))
    else:
        # Multi-class: probs is (N, K), labels is (N,)
        conf = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)
        correct = (preds == labels).astype(float)
        one_hot = np.zeros_like(probs)
        for i, l in enumerate(labels):
            one_hot[i, l] = 1.0
        brier = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
        # NLL
        chosen_probs = np.clip([probs[i, l] for i, l in enumerate(labels)], 1e-12, 1.0)
        nll = -np.mean(np.log(chosen_probs))

    # ECE computation
    bin_boundaries = np.linspace(0.5 if is_binary else (1.0 / probs.shape[-1] if not is_binary else 0.5), 1.0, n_bins + 1)
    ece = 0.0
    total = len(conf)
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (conf >= low) & (conf < high if i < n_bins - 1 else conf <= high)
        n_in = np.sum(in_bin)
        if n_in > 0:
            avg_conf = np.mean(conf[in_bin])
            avg_acc = np.mean(correct[in_bin])
            ece += (n_in / total) * np.abs(avg_acc - avg_conf)

    acc = np.mean(correct)
    mean_conf = np.mean(conf)
    return {
        "accuracy": float(acc),
        "brier": float(brier),
        "nll": float(nll),
        "ece": float(ece),
        "mean_confidence": float(mean_conf),
    }


def fit_binary_temperature(logits, labels, lr=0.01, max_iter=200):
    """Fit a single positive scalar T for binary sigmoid classification to minimize NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_t)
        scaled_logits = logits_t / T
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def fit_multiclass_temperature(logits_list, labels_list, lr=0.01, max_iter=200):
    """Fit a single positive scalar T for multi-class softmax classification to minimize NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    # Flatten into tensors (logits might have variable candidate counts across items)
    # We group by candidate count or compute row-by-row NLL
    tensors = [(torch.tensor(lg, dtype=torch.float32), int(lb)) for lg, lb in zip(logits_list, labels_list)]

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_t)
        total_loss = 0.0
        for lg, lb in tensors:
            scaled = lg / T
            log_p = F.log_softmax(scaled, dim=-1)
            total_loss = total_loss - log_p[lb]
        loss = total_loss / len(tensors)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def collect_validation_logits(checkpoint_dir, val_file, device="mps", batch_size=16):
    path = Path(checkpoint_dir)
    dev = device_for(device)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    config = AutoConfig.from_pretrained(path / "encoder", local_files_only=True)
    model = ThreeHeadEncoder(AutoModel.from_config(config, attn_implementation="sdpa"))
    model.load_state_dict(load_file(path / "model.safetensors"), strict=False)
    model.to(dev).eval()

    rows = read_rows(val_file)
    print(f"Encoding {len(rows)} validation rows...")
    encoded = [encode(r, tokenizer, 1024) for r in rows]
    loader = DataLoader(encoded, batch_size=batch_size, collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))

    data = {
        "noul": {"logits": [], "labels": []},
        "choice": {"logits": [], "labels": []},
        "score": {"logits": [], "labels": []},
    }

    # fp32 precision for calibration logit extraction (accuracy > speed on val set)
    autocast_ctx = contextlib.nullcontext()

    print("Extracting uncalibrated logits across validation set (fp32)...")
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, dev)
            with autocast_ctx:
                result = model(**batch)

            # Noul
            noul_valid = batch["labels"]["noul"] != -100
            if noul_valid.any():
                n_logits = result["noul"][noul_valid].cpu().float().tolist()
                n_labels = batch["labels"]["noul"][noul_valid].cpu().tolist()
                data["noul"]["logits"].extend(n_logits)
                data["noul"]["labels"].extend(n_labels)

            # Choice
            choice_valid = batch["labels"]["choice"] != -100
            if choice_valid.any():
                c_logits = result["choice"][choice_valid].cpu().float()
                c_labels = batch["labels"]["choice"][choice_valid].cpu().tolist()
                valid_mask = result["candidate_valid"][choice_valid].cpu()
                for i in range(len(c_labels)):
                    n_cands = int(valid_mask[i].sum().item())
                    data["choice"]["logits"].append(c_logits[i, :n_cands].tolist())
                    data["choice"]["labels"].append(int(c_labels[i]))

            # Score
            score_valid = batch["labels"]["score"] != -100
            if score_valid.any():
                s_logits = result["score"][score_valid].cpu().float()
                s_labels = batch["labels"]["score"][score_valid].cpu().tolist()
                valid_mask = result["candidate_valid"][score_valid].cpu()
                for i in range(len(s_labels)):
                    n_cands = int(valid_mask[i].sum().item())
                    data["score"]["logits"].append(s_logits[i, :n_cands].tolist())
                    data["score"]["labels"].append(int(s_labels[i]))

    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/dev-0.4b", help="Source checkpoint directory")
    parser.add_argument("--val-data", default="data/validation_refine.jsonl")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--apply-to-checkpoint", action="store_true", help="Write temperatures into run.json")
    args = parser.parse_args()

    data = collect_validation_logits(args.checkpoint, args.val_data, device=args.device)

    print("\n" + "=" * 70)
    print("           TEMPERATURE SCALING (GUO ET AL. 2017) OPTIMIZATION")
    print("=" * 70)

    results = {}
    temperatures = {}

    for task in ("noul", "choice", "score"):
        lgs = data[task]["logits"]
        lbs = data[task]["labels"]
        n_samples = len(lbs)
        if n_samples == 0:
            continue

        print(f"\n[{task.upper()}] ({n_samples} validation samples)")

        # 1. Uncalibrated baseline metrics (T = 1.0)
        if task == "noul":
            probs_uncal = 1.0 / (1.0 + np.exp(-np.array(lgs)))
            base_metrics = compute_ece_and_brier(probs_uncal, np.array(lbs), is_binary=True)
            T_opt = fit_binary_temperature(lgs, lbs)
            probs_cal = 1.0 / (1.0 + np.exp(-np.array(lgs) / T_opt))
            cal_metrics = compute_ece_and_brier(probs_cal, np.array(lbs), is_binary=True)
        else:
            # Multi-class
            probs_uncal_list = [np.exp(lg - np.max(lg)) / np.sum(np.exp(lg - np.max(lg))) for lg in lgs]
            # To compute metrics, find max cands
            max_c = max(len(lg) for lg in lgs)
            p_matrix_uncal = np.zeros((len(lgs), max_c))
            for i, p in enumerate(probs_uncal_list):
                p_matrix_uncal[i, :len(p)] = p
            base_metrics = compute_ece_and_brier(p_matrix_uncal, np.array(lbs), is_binary=False)

            T_opt = fit_multiclass_temperature(lgs, lbs)

            probs_cal_list = [np.exp((lg - np.max(lg)) / T_opt) / np.sum(np.exp((lg - np.max(lg)) / T_opt)) for lg in lgs]
            p_matrix_cal = np.zeros((len(lgs), max_c))
            for i, p in enumerate(probs_cal_list):
                p_matrix_cal[i, :len(p)] = p
            cal_metrics = compute_ece_and_brier(p_matrix_cal, np.array(lbs), is_binary=False)

        temperatures[task] = round(T_opt, 4)
        results[task] = {"before": base_metrics, "after": cal_metrics, "temperature": T_opt}

        print(f"  Fitted Temperature T_{task}: {T_opt:.4f}")
        print(f"  Accuracy:         {base_metrics['accuracy']:.4f} -> {cal_metrics['accuracy']:.4f} (100% Invariant)")
        print(f"  NLL Loss:         {base_metrics['nll']:.4f} -> {cal_metrics['nll']:.4f} (Delta: {cal_metrics['nll'] - base_metrics['nll']:.4f})")
        print(f"  Brier Score:      {base_metrics['brier']:.4f} -> {cal_metrics['brier']:.4f} (Delta: {cal_metrics['brier'] - base_metrics['brier']:.4f})")
        print(f"  ECE (Cal. Error): {base_metrics['ece']:.4f} -> {cal_metrics['ece']:.4f} (Delta: {cal_metrics['ece'] - base_metrics['ece']:.4f})")
        print(f"  Mean Confidence:  {base_metrics['mean_confidence']:.4f} -> {cal_metrics['mean_confidence']:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY OF OPTIMAL READOUT TEMPERATURES:")
    print(json.dumps(temperatures, indent=2))
    print("=" * 70)

    if args.apply_to_checkpoint:
        chk = Path(args.checkpoint)
        run_file = chk / "run.json"
        if run_file.exists():
            meta = json.loads(run_file.read_text())
            meta["temperatures"] = temperatures
            meta["calibration_method"] = "temperature_scaling_guo_2017"
            meta["status"] = "calibrated"
            meta["calibration_metrics"] = results
            run_file.write_text(json.dumps(meta, indent=2))
            print(f"\nSuccessfully updated {run_file} with optimal temperatures!")


if __name__ == "__main__":
    main()
