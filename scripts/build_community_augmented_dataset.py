"""Build Phase 3 Community-Augmented & Contrastive Dataset.

Augments the existing 21,601 multi-task rows with:
1. mteb/banking77: 3,000 train / 600 val (Choice: 77 categories)
2. Yelp/yelp_review_full: 3,000 train / 500 val (Score: 5-level ordinal)
3. Contrastive Code Mutations: ~1,000 adversarial hard negative code pairs (Noul/Score)

Outputs:
- data/train_v3.jsonl
- data/validation_v3.jsonl
"""
import ast
import json
import random
import re
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from dev.data import SCHEMA_VERSION, validate

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)


def mutate_python_code(code_str):
    """Introduce subtle adversarial logic mutations to create hard negatives."""
    mutations = []

    # 1. Flip comparison operators
    cmp_flips = {
        " == ": " != ",
        " != ": " == ",
        " < ": " >= ",
        " > ": " <= ",
        " <= ": " > ",
        " >= ": " < ",
        " is not ": " is ",
        " in ": " not in ",
    }
    for op, repl in cmp_flips.items():
        if op in code_str:
            mutations.append(code_str.replace(op, repl, 1))

    # 2. Invert boolean constants
    if "return True" in code_str:
        mutations.append(code_str.replace("return True", "return False", 1))
    elif "return False" in code_str:
        mutations.append(code_str.replace("return False", "return True", 1))

    # 3. Off-by-one errors in indexing / lengths
    if "len(" in code_str and "+" not in code_str:
        mutations.append(re.sub(r'len\(([^)]+)\)', r'len(\1) + 1', code_str, count=1))
    if "[0]" in code_str:
        mutations.append(code_str.replace("[0]", "[1]", 1))
    if "[-1]" in code_str:
        mutations.append(code_str.replace("[-1]", "[-2]", 1))

    # 4. Invert return condition
    if "return not " in code_str:
        mutations.append(code_str.replace("return not ", "return ", 1))
    elif "return " in code_str and "return not " not in code_str:
        # Try inserting 'not' if boolean-like
        mutations.append(re.sub(r'return\s+(is_|has_|can_)', r'return not \1', code_str, count=1))

    valid_mutations = [m for m in mutations if m != code_str]
    return valid_mutations[0] if valid_mutations else None


def build_banking77_rows(train_limit=3000, val_limit=600):
    """Choice: 77 banking intent categories."""
    print(f"Fetching Banking77 (train: {train_limit}, val: {val_limit})...")
    p_train = hf_hub_download("mteb/banking77", "data/train-00000-of-00001.parquet", repo_type="dataset")
    p_test = hf_hub_download("mteb/banking77", "data/test-00000-of-00001.parquet", repo_type="dataset")

    t_train = pq.read_table(p_train).to_pydict()
    t_test = pq.read_table(p_test).to_pydict()

    unique_labels = sorted(list(set(t_train["label_text"])))
    label_to_idx = {l: idx for idx, l in enumerate(unique_labels)}
    criteria = [l.replace("_", " ") for l in unique_labels]

    instructions = "Classify the customer banking request into the correct intent category."

    train_rows, val_rows = [], []

    # Train split
    for i in range(min(len(t_train["text"]), train_limit)):
        text = t_train["text"][i].strip()
        label_str = t_train["label_text"][i]
        label_idx = label_to_idx[label_str]
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"banking_train_{i}",
            "group": f"banking_train_grp_{i % 100}",
            "command": "",
            "state": text,
            "question": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria
            },
            "label": label_idx,
            "split": "train",
            "domain": "intent_routing"
        }
        validate(row)
        train_rows.append(row)

    # Val split
    for i in range(min(len(t_test["text"]), val_limit)):
        text = t_test["text"][i].strip()
        label_str = t_test["label_text"][i]
        label_idx = label_to_idx[label_str]
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"banking_val_{i}",
            "group": f"banking_val_grp_{i % 50}",
            "command": "",
            "state": text,
            "question": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria
            },
            "label": label_idx,
            "split": "development",
            "domain": "intent_routing"
        }
        validate(row)
        val_rows.append(row)

    print(f"  Generated {len(train_rows)} Banking77 train rows, {len(val_rows)} val rows.")
    return train_rows, val_rows


def build_yelp_rows(train_limit=3000, val_limit=500):
    """Score: 5-level star rating ordinal rubric."""
    print(f"Fetching Yelp Reviews (train: {train_limit}, val: {val_limit})...")
    p_train = hf_hub_download("Yelp/yelp_review_full", "yelp_review_full/train-00000-of-00001.parquet", repo_type="dataset")
    p_test = hf_hub_download("Yelp/yelp_review_full", "yelp_review_full/test-00000-of-00001.parquet", repo_type="dataset")

    t_train = pq.read_table(p_train).to_pydict()
    t_test = pq.read_table(p_test).to_pydict()

    criteria = [
        "1 star: Terrible, worst experience, strong negative",
        "2 stars: Poor, dissatisfied, below average",
        "3 stars: Average, acceptable, mixed feelings",
        "4 stars: Good, satisfied, pleasant experience",
        "5 stars: Excellent, highly recommended, outstanding"
    ]
    instructions = "Rate the customer sentiment of this review on a 1 to 5 scale."

    train_rows, val_rows = [], []

    for i in range(min(len(t_train["text"]), train_limit)):
        text = t_train["text"][i].strip()[:1000]
        label = int(t_train["label"][i])
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"yelp_train_{i}",
            "group": f"yelp_train_grp_{i % 100}",
            "command": "",
            "state": text,
            "question": {
                "type": "score",
                "instructions": instructions,
                "criteria": criteria
            },
            "label": label,
            "split": "train",
            "domain": "sentiment_scoring"
        }
        validate(row)
        train_rows.append(row)

    for i in range(min(len(t_test["text"]), val_limit)):
        text = t_test["text"][i].strip()[:1000]
        label = int(t_test["label"][i])
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"yelp_val_{i}",
            "group": f"yelp_val_grp_{i % 50}",
            "command": "",
            "state": text,
            "question": {
                "type": "score",
                "instructions": instructions,
                "criteria": criteria
            },
            "label": label,
            "split": "development",
            "domain": "sentiment_scoring"
        }
        validate(row)
        val_rows.append(row)

    print(f"  Generated {len(train_rows)} Yelp train rows, {len(val_rows)} val rows.")
    return train_rows, val_rows


def build_contrastive_code_rows():
    """Generate adversarial contrastive mutations from CodeSearchNet pairs."""
    pairs_file = ROOT / "data/teacher/semantic-pairs.jsonl"
    if not pairs_file.exists():
        print("Semantic pairs file not found, skipping contrastive mutations.")
        return [], []

    print("Generating Adversarial Contrastive Code Mutations...")
    records = [json.loads(line) for line in pairs_file.read_text().splitlines() if line.strip()]

    score_criteria = [
        "No relevance or incorrect / buggy logic",
        "Partially relevant context",
        "Correct implementation matching requested functionality"
    ]

    train_rows, val_rows = [], []
    idx = 0
    for r in records:
        code = r["state"].strip()
        query = r["query"].strip()
        group = r.get("group", f"code_grp_{idx % 40}")
        is_val = r.get("split") == "development"

        # 1. Mutate code to create adversarial hard negative
        mutated_code = mutate_python_code(code)
        if not mutated_code:
            continue

        # Add Noul negative (the mutated code has a bug / incorrect logic)
        noul_neg = {
            "schema_version": SCHEMA_VERSION,
            "id": f"contrastive_noul_neg_{idx}",
            "group": f"{group}_contrastive",
            "command": "git grep",
            "state": mutated_code[:1800],
            "question": {
                "type": "noul",
                "instructions": f"Does this code correctly implement the requested functionality: {query}?"
            },
            "label": 0,  # Negative!
            "split": "development" if is_val else "train",
            "domain": "contrastive_code"
        }

        # Add Score negative (mutated code receives score 0)
        score_neg = {
            "schema_version": SCHEMA_VERSION,
            "id": f"contrastive_score_neg_{idx}",
            "group": f"{group}_contrastive",
            "command": "git grep",
            "state": mutated_code[:1800],
            "question": {
                "type": "score",
                "instructions": f"How closely does this code match the requested functionality: {query}?",
                "criteria": score_criteria
            },
            "label": 0,  # 0: Buggy / incorrect logic
            "split": "development" if is_val else "train",
            "domain": "contrastive_code"
        }

        target_list = val_rows if is_val else train_rows
        for row in (noul_neg, score_neg):
            try:
                validate(row)
                target_list.append(row)
            except Exception:
                pass
        idx += 1

    print(f"  Generated {len(train_rows)} contrastive train rows, {len(val_rows)} val rows.")
    return train_rows, val_rows


def main():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    # 1. Load existing Phase 2 multi-task rows
    existing_train_path = ROOT / "data/train.jsonl"
    existing_val_path = ROOT / "data/validation.jsonl"
    print(f"Loading existing Phase 2 multi-task dataset...")
    existing_train = [json.loads(l) for l in existing_train_path.read_text().splitlines() if l.strip()]
    existing_val = [json.loads(l) for l in existing_val_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(existing_train)} existing train rows, {len(existing_val)} val rows.")

    # 2. Build Banking77
    bank_train, bank_val = build_banking77_rows(train_limit=3000, val_limit=600)

    # 3. Build Yelp
    yelp_train, yelp_val = build_yelp_rows(train_limit=3000, val_limit=500)

    # 4. Build Contrastive Code Mutations
    contrast_train, contrast_val = build_contrastive_code_rows()

    # 5. Combine and filter
    all_train = existing_train + bank_train + yelp_train + contrast_train
    all_val = existing_val + bank_val + yelp_val + contrast_val

    # Anti-contamination guarantee: ensure group disjointness
    train_groups = {r["group"] for r in all_train}
    val_groups = {r["group"] for r in all_val}
    overlap = train_groups & val_groups
    if overlap:
        print(f"WARNING: Resolving {len(overlap)} overlapping groups between train and val...")
        all_val = [r for r in all_val if r["group"] not in train_groups]

    print(f"\nVerifying sequence lengths (<= 1024 tokens)...")
    valid_train, valid_val = [], []
    for r in all_train:
        # Quick token length check
        tok_len = len(tokenizer.encode(r["state"][:3000]))
        if tok_len <= 1024:
            valid_train.append(r)

    for r in all_val:
        tok_len = len(tokenizer.encode(r["state"][:3000]))
        if tok_len <= 1024:
            valid_val.append(r)

    print(f"\nFinal Dataset Size:")
    print(f"  Train: {len(valid_train)} samples")
    print(f"  Validation: {len(valid_val)} samples")
    print(f"  Total: {len(valid_train) + len(valid_val)} samples")

    train_out = ROOT / "data/train_v3.jsonl"
    val_out = ROOT / "data/validation_v3.jsonl"

    with open(train_out, "w") as f:
        for r in valid_train:
            f.write(json.dumps(r) + "\n")

    with open(val_out, "w") as f:
        for r in valid_val:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {train_out} ({train_out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Wrote {val_out} ({val_out.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
