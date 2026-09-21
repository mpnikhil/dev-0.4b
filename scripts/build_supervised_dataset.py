"""Assemble multi-task supervised training and validation datasets from teacher predictions and semantic pairs.

Strictly adheres to schema_version: 2 with zero repository leakage between train and validation splits.
"""
import json
from collections import Counter
from pathlib import Path
from dev.data import SCHEMA_VERSION, validate

ROOT = Path(__file__).resolve().parents[1]

RESERVED_BENCHMARK_REPOS = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
}


def build_semantic_rows(pair):
    q_text = pair["query"].strip()
    role = pair["pair_role"]
    is_positive = role == "paired_documentation"
    state = pair["state"]
    group = pair["group"]
    split = pair["split"]
    cmd = "git grep -n -C 5"

    rows = []

    # 1. Noul
    noul_q = {
        "type": "noul",
        "instructions": f"Does this code implement the following functionality: {q_text}?"
    }
    rows.append({
        "schema_version": SCHEMA_VERSION,
        "id": f"sem_{pair['id']}_noul",
        "group": group,
        "command": cmd,
        "state": state,
        "question": noul_q,
        "label": 1 if is_positive else 0,
        "split": split,
        "domain": "code_search"
    })

    # 2. Score
    score_q = {
        "type": "score",
        "instructions": f"How closely does this code match the requested functionality: {q_text}?",
        "criteria": [
            "No relevance / shares keywords only",
            "Partially relevant context",
            "Direct implementation of requested functionality"
        ]
    }
    rows.append({
        "schema_version": SCHEMA_VERSION,
        "id": f"sem_{pair['id']}_score",
        "group": group,
        "command": cmd,
        "state": state,
        "question": score_q,
        "label": 2 if is_positive else 0,
        "split": split,
        "domain": "code_search"
    })

    # 3. Choice
    choice_q = {
        "type": "choice",
        "instructions": f"Evaluate this code against the search query: {q_text}",
        "criteria": [
            "Direct implementation of the query",
            "Lexical overlap but different implementation",
            "Unrelated code"
        ]
    }
    rows.append({
        "schema_version": SCHEMA_VERSION,
        "id": f"sem_{pair['id']}_choice",
        "group": group,
        "command": cmd,
        "state": state,
        "question": choice_q,
        "label": 0 if is_positive else 1,
        "split": split,
        "domain": "code_search"
    })

    return rows


def main():
    pred_path = ROOT / "data/teacher/agy_predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing teacher predictions: {pred_path}")

    predictions = {}
    for line in pred_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("valid"):
                predictions[row["id"]] = row

    print(f"Loaded {len(predictions)} valid teacher predictions.")

    # Load trace blocks
    blocks = {}
    for split in ["development", "train"]:
        split_path = ROOT / f"data/teacher/{split}.blocks.jsonl"
        if split_path.exists():
            for line in split_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    blocks[r["id"]] = r

    trace_rows = []
    dropped_unlabeled = 0
    for block_id, pred in predictions.items():
        if block_id not in blocks:
            continue
        block = blocks[block_id]
        if block["group"].lower() in RESERVED_BENCHMARK_REPOS:
            continue

        labels = pred["labels"]
        for q, label in zip(block["questions"], labels):
            if label is None or label == -100:
                dropped_unlabeled += 1
                continue
            row = {
                "schema_version": SCHEMA_VERSION,
                "id": f"trace_{block_id}_{q['type']}",
                "group": block["group"],
                "command": block.get("command", ""),
                "state": block["state"],
                "question": q,
                "label": label,
                "split": block["split"],
                "domain": "test_logs",
                "quotes": pred.get("quotes", []),
                "evidence_byte_spans": pred.get("evidence_byte_spans", [])
            }
            validate(row)
            trace_rows.append(row)

    print(f"Assembled {len(trace_rows)} trace block rows (dropped {dropped_unlabeled} null labels).")

    # Load semantic pairs
    sem_path = ROOT / "data/teacher/semantic-pairs.jsonl"
    sem_rows = []
    if sem_path.exists():
        for line in sem_path.read_text().splitlines():
            if line.strip():
                pair = json.loads(line)
                if pair["group"].lower() in RESERVED_BENCHMARK_REPOS:
                    continue
                for r in build_semantic_rows(pair):
                    validate(r)
                    sem_rows.append(r)

    print(f"Assembled {len(sem_rows)} semantic code search rows.")

    all_rows = trace_rows + sem_rows

    # Partition by split and group
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "development"]

    train_groups = {r["group"].lower() for r in train_rows}
    val_groups = {r["group"].lower() for r in val_rows}
    leakage = train_groups & val_groups
    if leakage:
        # Move any overlapping groups from validation to train or exclude them
        print(f"Resolving {len(leakage)} overlapping groups by removing from validation: {leakage}")
        val_rows = [r for r in val_rows if r["group"].lower() not in leakage]
        val_groups = {r["group"].lower() for r in val_rows}
        assert len(train_groups & val_groups) == 0, "Zero repository leakage violated!"

    # Write datasets
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows))
    (data_dir / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val_rows))

    summary = {
        "train": {
            "total_rows": len(train_rows),
            "repositories": len(train_groups),
            "by_type": dict(Counter(r["question"]["type"] for r in train_rows)),
            "by_domain": dict(Counter(r.get("domain", "unknown") for r in train_rows)),
        },
        "validation": {
            "total_rows": len(val_rows),
            "repositories": len(val_groups),
            "by_type": dict(Counter(r["question"]["type"] for r in val_rows)),
            "by_domain": dict(Counter(r.get("domain", "unknown") for r in val_rows)),
        },
        "reserved_repos_excluded": sorted(RESERVED_BENCHMARK_REPOS)
    }

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "supervised-dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {data_dir/'train.jsonl'} and {data_dir/'validation.jsonl'}")


if __name__ == "__main__":
    main()
