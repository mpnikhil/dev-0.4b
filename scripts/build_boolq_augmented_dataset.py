"""Build BoolQ-Augmented Refinement Dataset.

This script constructs the dataset for unifying to a Single Universal Choice Head
and closing the BoolQ benchmark gap:
1. Ingests all 9,427 official Google BoolQ training examples mapped as candidate choice:
   criteria: ["No", "Yes"], label: 0 (No) / 1 (Yes).
2. Curates high-signal multi-task anchors from train_v3.jsonl:
   - Banking77: 3,000 rows (77-way choice)
   - Yelp Sentiment: 3,000 rows (5-star score)
   - CodeSearchNet: 1,500 rows (code retrieval)
   - Contrastive Code Mutations: 652 rows (adversarial logic)
   - SWE-agent test_logs: 1,471 rows (tool selection / failure triage)
   - HelpSteer2 / ARC anchors: ~3,000 rows (reasoning / alignment)
3. Enforces token length <= 1024 and disjoint groups between train and validation.
4. Outputs:
   - data/train_refine.jsonl (~22,500 rows)
   - data/validation_refine.jsonl (~4,285 rows)
"""
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from dev.data import SCHEMA_VERSION, validate

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)


def fetch_all_boolq_train():
    """Download full official Google BoolQ train split (9,427 samples)."""
    print("Downloading full official Google BoolQ training set (9,427 samples)...")
    p = hf_hub_download(repo_id="google/boolq", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(p)
    d = table.to_pydict()

    rows = []
    n = len(d["question"])
    print(f"  Downloaded {n} raw BoolQ train samples.")
    for i in range(n):
        q_text = d["question"][i].strip()
        ans = d["answer"][i]
        passage = d["passage"][i].strip()
        if not passage or not q_text:
            continue
        if len(passage) > 1800:
            passage = passage[:1800]

        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"boolq_full_{i}",
            "group": f"boolq_train_grp_{i % 200}",
            "command": "",
            "state": passage,
            "question": {
                "type": "noul",
                "instructions": f"Does this passage answer the question with yes: {q_text}?",
                "criteria": ["No", "Yes"]
            },
            "label": 1 if ans else 0,
            "split": "train",
            "domain": "reading_comprehension"
        }
        try:
            validate(row)
            rows.append(row)
        except Exception:
            continue
    print(f"  Valid BoolQ train rows formatted: {len(rows)}")
    return rows


def build_refinement_dataset():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    # 1. Fetch full BoolQ train
    boolq_train = fetch_all_boolq_train()

    # 2. Load existing train_v3 anchors
    v3_train_path = ROOT / "data/train_v3.jsonl"
    print(f"\nLoading anchor rows from {v3_train_path}...")
    v3_train = [json.loads(l) for l in v3_train_path.read_text().splitlines() if l.strip()]

    # Filter by domain for curated anchors
    anchors = []
    domain_counts = {}
    limits = {
        "intent_routing": 3000,          # Banking77
        "sentiment_scoring": 3000,       # Yelp
        "code_search": 1500,             # CodeSearchNet
        "contrastive_code": 1000,        # Adversarial mutations
        "test_logs": 1500,               # SWE-agent
        "response_evaluation": 1500,     # HelpSteer2
        "multi_choice_reasoning": 1500,  # ARC
        "context_answerability": 1500,   # SQuAD
        "agent_intent_routing": 1500,    # MASSIVE
    }

    # Shuffle v3 rows with fixed seed
    shuffled_v3 = list(v3_train)
    random.shuffle(shuffled_v3)

    for r in shuffled_v3:
        dom = r.get("domain", "unknown")
        if dom == "reading_comprehension":
            continue  # Replace old BoolQ subset with the full 9,427 BoolQ dataset
        max_allowed = limits.get(dom, 1000)
        curr = domain_counts.get(dom, 0)
        if curr < max_allowed:
            # If noul row, ensure criteria is explicit ["No", "Yes"]
            if r["question"]["type"] == "noul" and not r["question"].get("criteria"):
                r["question"]["criteria"] = ["No", "Yes"]
            anchors.append(r)
            domain_counts[dom] = curr + 1

    print(f"Curated {len(anchors)} anchor rows across domains:")
    for dom, cnt in sorted(domain_counts.items()):
        print(f"  - {dom}: {cnt}")

    # Combine full BoolQ train with curated anchors
    all_train = boolq_train + anchors
    random.shuffle(all_train)

    # 3. Load validation rows
    v3_val_path = ROOT / "data/validation_v3.jsonl"
    print(f"\nLoading validation rows from {v3_val_path}...")
    v3_val = [json.loads(l) for l in v3_val_path.read_text().splitlines() if l.strip()]
    for r in v3_val:
        if r["question"]["type"] == "noul" and not r["question"].get("criteria"):
            r["question"]["criteria"] = ["No", "Yes"]

    # Ensure group disjointness
    train_groups = {r["group"] for r in all_train}
    val_groups = {r["group"] for r in v3_val}
    overlap = train_groups & val_groups
    if overlap:
        print(f"Resolving {len(overlap)} overlapping groups between train and validation...")
        v3_val = [r for r in v3_val if r["group"] not in train_groups]

    # Verify token lengths <= 1024
    print("\nVerifying token lengths (<= 1024)...")
    valid_train = []
    for r in all_train:
        tok_len = len(tokenizer.encode(r["state"][:3000]))
        if tok_len <= 1024:
            valid_train.append(r)

    valid_val = []
    for r in v3_val:
        tok_len = len(tokenizer.encode(r["state"][:3000]))
        if tok_len <= 1024:
            valid_val.append(r)

    print(f"\nRefinement Dataset Summary:")
    print(f"  Train: {len(valid_train)} rows")
    print(f"  Validation: {len(valid_val)} rows")

    train_out = ROOT / "data/train_refine.jsonl"
    val_out = ROOT / "data/validation_refine.jsonl"

    with open(train_out, "w") as f:
        for r in valid_train:
            f.write(json.dumps(r) + "\n")

    with open(val_out, "w") as f:
        for r in valid_val:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {train_out} ({train_out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Wrote {val_out} ({val_out.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    build_refinement_dataset()
