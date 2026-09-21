"""Evaluate trained Jev cross-encoder on Nimble public benchmark tasks (Choice, Noul, Score).

Supports testing directly against public evaluation datasets.
"""
import argparse
import json
import math
from pathlib import Path
from dev.inference import Predictor

ROOT = Path(__file__).resolve().parents[1]


def evaluate_boolq(predictor, records):
    """Evaluate Noul head on BoolQ yes/no questions."""
    correct = 0
    brier_sum = 0.0
    instructions = "Does the passage answer the question with yes? Use only what the passage states or directly implies."

    for row in records:
        passage = row.get("passage", "")
        question = row.get("question", "")
        gold_bool = row.get("answer")
        state = f"Question: {question}\n\nPassage:\n{passage}"

        ans = predictor.answer(
            state=state,
            questions={"decision": {"type": "noul", "instructions": instructions}}
        )
        prob = ans["decision"]["probability_yes"]
        pred_bool = prob >= 0.5
        if pred_bool == gold_bool:
            correct += 1
        brier_sum += (prob - float(gold_bool)) ** 2

    n = len(records)
    return {
        "count": n,
        "accuracy": correct / max(n, 1),
        "brier_score": brier_sum / max(n, 1)
    }


def evaluate_vitaminc(predictor, records):
    """Evaluate Choice head on VitaminC fact verification (3 options)."""
    criteria = [
        "The evidence states the claim, or the claim follows directly from the evidence.",
        "The evidence states the opposite of the claim, or the claim is directly contradicted by it.",
        "The evidence neither establishes nor contradicts the claim. Settling it would need a fact the evidence does not supply."
    ]
    labels_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    instructions = "Decide how the evidence bears on the claim. Judge only from the evidence text, and not from outside knowledge about the subject."

    correct = 0
    nll_sum = 0.0

    for row in records:
        evidence = row.get("evidence", "")
        claim = row.get("claim", "")
        gold_str = row.get("label", "")
        if gold_str not in labels_map:
            continue
        gold_idx = labels_map[gold_str]
        state = f"Claim: {claim}\n\nEvidence:\n{evidence}"

        ans = predictor.answer(
            state=state,
            questions={"fact": {"type": "choice", "instructions": instructions, "criteria": criteria}}
        )
        probs = ans["fact"]["probabilities"]
        pred_idx = ans["fact"]["index"]
        if pred_idx == gold_idx:
            correct += 1
        p = max(probs[gold_idx], 1e-9)
        nll_sum += -math.log(p)

    n = len(records)
    return {
        "count": n,
        "accuracy": correct / max(n, 1),
        "mean_nll": nll_sum / max(n, 1)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    p.add_argument("--task", default="summary", choices=["summary", "boolq", "vitaminc"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="docs/nimble-benchmark-eval.json")
    args = p.parse_args()

    predictor = Predictor(args.checkpoint, device=args.device)
    print(f"Loaded predictor from {args.checkpoint} on {predictor.device}")

    # Inspect test predictions if available
    report = {
        "checkpoint": args.checkpoint,
        "device": str(predictor.device),
        "benchmarks": {}
    }

    out_file = ROOT / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report initialized at {out_file}")


if __name__ == "__main__":
    main()
