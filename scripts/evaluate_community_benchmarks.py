"""Evaluate Jev ModernBERT cross-encoder on standard open-source Jev community benchmarks.

Tests the three primitives:
1. Noul: google/boolq (Held-out validation set) -> Accuracy, Brier, ECE
2. Choice: mteb/banking77 (Standard test set, 77 categories) -> Top-1 Acc, Top-3 Recall, ECE
3. Score: Yelp review test set (5-level ordinal rating) -> Exact Acc, MAE, ECE

Outputs a unified report matching the format of Jared Palmer's Kev and community trackers.
"""
import argparse
import json
import math
import time
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from dev.inference import Predictor

ROOT = Path(__file__).resolve().parents[1]


def compute_ece(confidences, predictions, labels, num_bins=10):
    """Compute Expected Calibration Error (ECE) across B probability bins."""
    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = [
            j for j in range(n)
            if (bin_lower <= confidences[j] < bin_upper or (i == num_bins - 1 and confidences[j] == 1.0))
        ]
        if in_bin:
            bin_acc = sum(1 for j in in_bin if predictions[j] == labels[j]) / len(in_bin)
            bin_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
            ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)
    return ece


def evaluate_boolq(predictor, sample_limit=500):
    """Noul evaluation on google/boolq validation split."""
    print(f"\n[1/3] Benchmarking Noul on google/boolq (sample_limit={sample_limit})...")
    p = hf_hub_download(repo_id="google/boolq", filename="data/validation-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(p)
    d = table.to_pydict()

    n = min(len(d["question"]), sample_limit)
    correct = 0
    brier_sum = 0.0
    confidences = []
    predictions = []
    labels = []

    start_t = time.time()
    for i in range(n):
        q_text = d["question"][i]
        passage = d["passage"][i]
        gold_answer = bool(d["answer"][i])

        state = passage[:1800]
        ans = predictor.answer(
            state=state,
            questions={"val": {"type": "noul", "instructions": f"Does this passage answer the question with yes: {q_text}?"}}
        )
        prob_yes = ans["val"]["probability_yes"]
        pred_answer = prob_yes >= 0.5
        conf = prob_yes if pred_answer else (1.0 - prob_yes)

        if pred_answer == gold_answer:
            correct += 1
        brier_sum += (prob_yes - float(gold_answer)) ** 2

        confidences.append(conf)
        predictions.append(pred_answer)
        labels.append(gold_answer)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  Processed {i+1}/{n} BoolQ items ({time.time()-start_t:.1f}s) - Current Acc: {correct/(i+1):.3f}")

    acc = correct / n
    brier = brier_sum / n
    ece = compute_ece(confidences, predictions, labels)

    print(f"BoolQ Results -> Acc: {acc:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
    return {
        "dataset": "google/boolq (validation)",
        "task": "Noul (Boolean Decision)",
        "sample_count": n,
        "accuracy": acc,
        "brier_score": brier,
        "ece_10_bins": ece,
        "time_seconds": time.time() - start_t
    }


def evaluate_banking77(predictor, sample_limit=300):
    """Choice evaluation on mteb/banking77 test split (77 categories)."""
    print(f"\n[2/3] Benchmarking Choice on mteb/banking77 (sample_limit={sample_limit}, 77 classes)...")
    p = hf_hub_download(repo_id="mteb/banking77", filename="data/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(p)
    d = table.to_pydict()

    # Collect all unique label texts to build the 77 criteria list
    unique_labels = sorted(list(set(d["label_text"])))
    label_to_idx = {l: idx for idx, l in enumerate(unique_labels)}

    # Convert snake_case labels to natural readable criteria
    criteria = [l.replace("_", " ") for l in unique_labels]

    instructions = "Classify the customer banking request into the correct intent category."

    n = min(len(d["text"]), sample_limit)
    correct_top1 = 0
    correct_top3 = 0
    confidences = []
    predictions = []
    labels = []

    start_t = time.time()
    for i in range(n):
        user_query = d["text"][i]
        gold_label_text = d["label_text"][i]
        gold_idx = label_to_idx[gold_label_text]

        ans = predictor.answer(
            state=user_query,
            questions={"intent": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria
            }}
        )

        probs = ans["intent"]["probabilities"]
        pred_idx = ans["intent"]["index"]
        conf = probs[pred_idx]

        # Top 3 check
        top3_indices = sorted(range(len(probs)), key=lambda k: -probs[k])[:3]
        if pred_idx == gold_idx:
            correct_top1 += 1
        if gold_idx in top3_indices:
            correct_top3 += 1

        confidences.append(conf)
        predictions.append(pred_idx)
        labels.append(gold_idx)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Processed {i+1}/{n} Banking77 items ({time.time()-start_t:.1f}s) - Top-1: {correct_top1/(i+1):.3f}, Top-3: {correct_top3/(i+1):.3f}")

    top1_acc = correct_top1 / n
    top3_recall = correct_top3 / n
    ece = compute_ece(confidences, predictions, labels)

    print(f"Banking77 Results -> Top-1: {top1_acc:.4f} | Top-3 Recall: {top3_recall:.4f} | ECE: {ece:.4f}")
    return {
        "dataset": "mteb/banking77 (test)",
        "task": "Choice (77-way Routing)",
        "num_classes": len(criteria),
        "sample_count": n,
        "top1_accuracy": top1_acc,
        "top3_recall": top3_recall,
        "ece_10_bins": ece,
        "time_seconds": time.time() - start_t
    }


def evaluate_yelp(predictor, sample_limit=300):
    """Score evaluation on Yelp review test split (1-5 star ordinal grading)."""
    print(f"\n[3/3] Benchmarking Score on Yelp Reviews (sample_limit={sample_limit}, 5 levels)...")
    p = hf_hub_download(repo_id="Yelp/yelp_review_full", filename="yelp_review_full/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(p)
    d = table.to_pydict()

    criteria = [
        "1 star: Terrible, worst experience, strong negative",
        "2 stars: Poor, dissatisfied, below average",
        "3 stars: Average, acceptable, mixed feelings",
        "4 stars: Good, satisfied, pleasant experience",
        "5 stars: Excellent, highly recommended, outstanding"
    ]
    instructions = "Rate the customer sentiment of this review on a 1 to 5 scale."

    n = min(len(d["text"]), sample_limit)
    exact_correct = 0
    mae_sum = 0.0
    confidences = []
    predictions = []
    labels = []

    start_t = time.time()
    for i in range(n):
        review_text = d["text"][i][:1000]  # truncate long reviews to first 1k chars
        gold_label = int(d["label"][i])    # 0 to 4 in dataset

        ans = predictor.answer(
            state=review_text,
            questions={"rating": {
                "type": "score",
                "instructions": instructions,
                "criteria": criteria
            }}
        )

        pred_val = ans["rating"]["value"]
        probs = ans["rating"]["probabilities"]
        pred_idx = max(range(len(probs)), key=probs.__getitem__)
        conf = probs[pred_idx]

        if pred_idx == gold_label:
            exact_correct += 1
        mae_sum += abs(pred_val - gold_label)

        confidences.append(conf)
        predictions.append(pred_idx)
        labels.append(gold_label)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Processed {i+1}/{n} Yelp items ({time.time()-start_t:.1f}s) - Exact Acc: {exact_correct/(i+1):.3f}, MAE: {mae_sum/(i+1):.3f}")

    exact_acc = exact_correct / n
    mae = mae_sum / n
    ece = compute_ece(confidences, predictions, labels)

    print(f"Yelp Results -> Exact Acc: {exact_acc:.4f} | MAE: {mae:.4f} | ECE: {ece:.4f}")
    return {
        "dataset": "Yelp/yelp_review_full (test)",
        "task": "Score (5-level Ordinal Rating)",
        "sample_count": n,
        "exact_accuracy": exact_acc,
        "mean_absolute_error": mae,
        "ece_10_bins": ece,
        "time_seconds": time.time() - start_t
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/dev-0.4b", help="Model checkpoint directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--boolq-samples", type=int, default=300)
    p.add_argument("--banking-samples", type=int, default=200)
    p.add_argument("--yelp-samples", type=int, default=200)
    p.add_argument("--t-noul", type=float, default=None, help="Override noul temperature")
    p.add_argument("--t-choice", type=float, default=None, help="Override choice temperature")
    p.add_argument("--t-score", type=float, default=None, help="Override score temperature")
    p.add_argument("--output", default="docs/community-benchmark-results.json")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint} (device={args.device})...")
    predictor = Predictor(args.checkpoint, device=args.device)

    # Apply temperature overrides if supplied
    if any(t is not None for t in (args.t_noul, args.t_choice, args.t_score)):
        predictor.temperatures = dict(predictor.temperatures)
        if args.t_noul is not None:
            predictor.temperatures["noul"] = args.t_noul
        if args.t_choice is not None:
            predictor.temperatures["choice"] = args.t_choice
        if args.t_score is not None:
            predictor.temperatures["score"] = args.t_score

    print(f"Active temperatures: {predictor.temperatures}")
    print(f"Model ready on {predictor.device}!")

    results = {
        "model": "ModernBERT-large (399M)",
        "checkpoint": args.checkpoint,
        "device": str(predictor.device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "benchmarks": {}
    }

    if args.boolq_samples > 0:
        results["benchmarks"]["boolq"] = evaluate_boolq(predictor, sample_limit=args.boolq_samples)

    if args.banking_samples > 0:
        results["benchmarks"]["banking77"] = evaluate_banking77(predictor, sample_limit=args.banking_samples)

    if args.yelp_samples > 0:
        results["benchmarks"]["yelp"] = evaluate_yelp(predictor, sample_limit=args.yelp_samples)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n=======================================================")
    print(f"Community Benchmark Evaluation Complete! Report written to: {out_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
