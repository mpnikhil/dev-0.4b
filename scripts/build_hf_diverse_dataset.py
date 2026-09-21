"""Assemble ~24,000 diverse multi-task training and validation samples across Choice, Score, and Noul.

Draws from:
- Choice: Amazon Massive (15-30 dynamic intent options), AI2 ARC (4-5 options), SWE diagnostics
- Score: HelpSteer2 (5-level ordinal 0..4), Semantic Code Search (3-level 0..2), SWE logs
- Noul: BoolQ (yes/no reading comprehension), SQuAD 2.0 (context answerability), SWE verification
"""
import gzip
import json
import random
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

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


def build_boolq_rows(limit=4000):
    """Noul: Reading comprehension yes/no."""
    print(f"Fetching BoolQ (target {limit})...")
    p = hf_hub_download(repo_id="google/boolq", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(p)
    d = table.to_pydict()

    rows = []
    n = min(len(d["question"]), limit)
    for i in range(n):
        q_text = d["question"][i].strip()
        ans = d["answer"][i]
        passage = d["passage"][i].strip()
        if not passage or not q_text:
            continue
        # Truncate overly verbose passages to ~1500 chars to stay safely within 1024 tokens
        if len(passage) > 1800:
            passage = passage[:1800]

        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"boolq_{i}",
            "group": f"boolq_doc_{i % 100}",
            "command": "",
            "state": passage,
            "question": {
                "type": "noul",
                "instructions": f"Does this passage answer the question with yes: {q_text}?"
            },
            "label": 1 if ans else 0,
            "split": "train" if (i % 5 != 0) else "development",
            "domain": "reading_comprehension"
        }
        try:
            validate(row)
            rows.append(row)
        except Exception:
            continue
    print(f"  Generated {len(rows)} BoolQ Noul rows.")
    return rows


def build_squad_rows(limit=3500):
    """Noul: Context answerability (binary)."""
    print(f"Fetching SQuAD 2.0 (target {limit})...")
    p = hf_hub_download(repo_id="rajpurkar/squad_v2", filename="squad_v2/train-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(p)
    d = table.to_pydict()

    rows = []
    # Balance answerable vs unanswerable
    ans_rows, unans_rows = [], []
    for i in range(len(d["question"])):
        context = d["context"][i].strip()
        q_text = d["question"][i].strip()
        has_ans = len(d["answers"][i]["text"]) > 0
        if not context or not q_text:
            continue
        if len(context) > 1800:
            context = context[:1800]

        is_val = (i % 5 == 0)
        target_list = ans_rows if has_ans else unans_rows
        if len(target_list) < limit // 2:
            target_list.append({
                "schema_version": SCHEMA_VERSION,
                "id": f"squad2_{i}",
                "group": f"squad_{d['title'][i]}",
                "command": "",
                "state": context,
                "question": {
                    "type": "noul",
                    "instructions": f"Can the following question be answered directly from this text: {q_text}?"
                },
                "label": 1 if has_ans else 0,
                "split": "development" if is_val else "train",
                "domain": "context_answerability"
            })
        if len(ans_rows) >= limit // 2 and len(unans_rows) >= limit // 2:
            break

    for r in ans_rows + unans_rows:
        try:
            validate(r)
            rows.append(r)
        except Exception:
            continue
    print(f"  Generated {len(rows)} SQuAD 2.0 Noul rows.")
    return rows


def build_massive_rows(limit=3500):
    """Choice: Agent Intent & Tool Routing (Dynamic candidate criteria: 15-25 options per sample)."""
    print(f"Fetching Amazon Massive Intent (target {limit})...")
    p = hf_hub_download(repo_id="SetFit/amazon_massive_intent_en-US", filename="train.jsonl", repo_type="dataset")
    raw_lines = [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]

    # Collect all unique intents
    all_intents = sorted(list({r["label_text"] for r in raw_lines}))
    intent_desc = {
        name: f"Action: {name.replace('_', ' ')}" for name in all_intents
    }

    rows = []
    random.seed(42)
    selected = raw_lines[:limit]
    for i, r in enumerate(selected):
        true_intent = r["label_text"]
        user_text = r["text"].strip()
        if not user_text or true_intent not in intent_desc:
            continue

        # Sample 15 to 20 distractors + the true intent
        k_options = random.randint(12, 20)
        distractors = [x for x in all_intents if x != true_intent]
        sampled_distractors = random.sample(distractors, k_options - 1)
        criteria_names = sampled_distractors + [true_intent]
        random.shuffle(criteria_names)

        criteria = [intent_desc[c] for c in criteria_names]
        label_idx = criteria_names.index(true_intent)

        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"massive_{i}",
            "group": f"massive_intent_{true_intent.split('_')[0]}",
            "command": "agent.route_intent",
            "state": f"User Request: {user_text}",
            "question": {
                "type": "choice",
                "instructions": "Select the appropriate intent action for this user utterance:",
                "criteria": criteria
            },
            "label": label_idx,
            "split": "development" if (i % 5 == 0) else "train",
            "domain": "agent_intent_routing"
        }
        try:
            validate(row)
            rows.append(row)
        except Exception:
            continue

    print(f"  Generated {len(rows)} Massive Choice rows (12..20 options each).")
    return rows


def build_arc_rows(limit=3000):
    """Choice: Multiple-Choice Reasoning (4..5 options)."""
    print(f"Fetching AI2 ARC Challenge & Easy (target {limit})...")
    p_chal = hf_hub_download(repo_id="allenai/ai2_arc", filename="ARC-Challenge/train-00000-of-00001.parquet", repo_type="dataset")
    p_easy = hf_hub_download(repo_id="allenai/ai2_arc", filename="ARC-Easy/train-00000-of-00001.parquet", repo_type="dataset")

    tables = [pq.read_table(p_chal).to_pydict(), pq.read_table(p_easy).to_pydict()]
    rows = []
    idx = 0
    for d in tables:
        for i in range(len(d["question"])):
            if len(rows) >= limit:
                break
            q_text = d["question"][i].strip()
            ans_key = d["answerKey"][i].strip()
            choices = d["choices"][i]
            texts = [t.strip() for t in choices["text"]]
            labels = [l.strip() for l in choices["label"]]

            # Ensure valid criteria and answer mapping
            if ans_key not in labels:
                continue
            label_idx = labels.index(ans_key)
            if not 2 <= len(texts) <= 10 or any(not t for t in texts) or len(set(texts)) != len(texts):
                continue

            row = {
                "schema_version": SCHEMA_VERSION,
                "id": f"arc_{idx}",
                "group": f"arc_q_{idx % 100}",
                "command": "",
                "state": f"Problem Context: {q_text}",
                "question": {
                    "type": "choice",
                    "instructions": "Select the correct scientific conclusion or answer:",
                    "criteria": texts
                },
                "label": label_idx,
                "split": "development" if (idx % 5 == 0) else "train",
                "domain": "multi_choice_reasoning"
            }
            idx += 1
            try:
                validate(row)
                rows.append(row)
            except Exception:
                continue

    print(f"  Generated {len(rows)} ARC Choice rows (4..5 options each).")
    return rows


def build_helpsteer_rows(limit=4500):
    """Score: HelpSteer2 (5-level ordinal scale 0..4 for Helpfulness and Correctness)."""
    print(f"Fetching HelpSteer2 (target {limit})...")
    p = hf_hub_download(repo_id="nvidia/HelpSteer2", filename="train.jsonl.gz", repo_type="dataset")

    criteria_helpfulness = [
        "Level 0: Completely unhelpful, irrelevant, or refusal without cause",
        "Level 1: Minimally helpful, shallow, or misses key context",
        "Level 2: Partially helpful with moderate relevance",
        "Level 3: Mostly helpful, addresses main aspects accurately",
        "Level 4: Exceptionally helpful, thorough, well-structured, and complete"
    ]
    criteria_correctness = [
        "Level 0: Entirely incorrect, misleading, or factual hallucinations",
        "Level 1: Contains major factual errors with few accurate elements",
        "Level 2: Partially correct but contains noticeable inaccuracies",
        "Level 3: Mostly correct with only minor, non-critical inaccuracies",
        "Level 4: Completely factually correct, rigorous, and verified"
    ]

    rows = []
    with gzip.open(p, "rt") as f:
        idx = 0
        for line in f:
            if len(rows) >= limit:
                break
            r = json.loads(line)
            prompt = r["prompt"].strip()
            resp = r["response"].strip()
            help_score = int(r.get("helpfulness", -1))
            corr_score = int(r.get("correctness", -1))

            if not prompt or not resp or not (0 <= help_score <= 4) or not (0 <= corr_score <= 4):
                continue

            state = f"Instruction:\n{prompt}\n\nCandidate Response:\n{resp}"
            if len(state) > 1200:
                state = state[:1200]

            is_val = (idx % 5 == 0)
            # Alternating between helpfulness and correctness
            if idx % 2 == 0:
                q_obj = {
                    "type": "score",
                    "instructions": "Evaluate the helpfulness of this response on a 5-level ordinal scale:",
                    "criteria": criteria_helpfulness
                }
                label = help_score
            else:
                q_obj = {
                    "type": "score",
                    "instructions": "Evaluate the factual correctness of this response on a 5-level ordinal scale:",
                    "criteria": criteria_correctness
                }
                label = corr_score

            row = {
                "schema_version": SCHEMA_VERSION,
                "id": f"helpsteer_{idx}",
                "group": f"helpsteer_item_{idx % 120}",
                "command": "agent.evaluate_response",
                "state": state,
                "question": q_obj,
                "label": label,
                "split": "development" if is_val else "train",
                "domain": "response_evaluation"
            }
            idx += 1
            try:
                validate(row)
                rows.append(row)
            except Exception:
                continue

    print(f"  Generated {len(rows)} HelpSteer2 Score rows (5 levels 0..4).")
    return rows


def load_swe_and_code_search_rows():
    """Load trace blocks and semantic code search pairs from Phase 1."""
    print("Loading SWE-bench trace blocks and CodeSearchNet semantic pairs...")
    pred_path = ROOT / "data/teacher/agy_predictions.jsonl"
    predictions = {}
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("valid"):
                    predictions[row["id"]] = row

    blocks = {}
    for split in ["development", "train"]:
        split_path = ROOT / f"data/teacher/{split}.blocks.jsonl"
        if split_path.exists():
            for line in split_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    blocks[r["id"]] = r

    trace_rows = []
    for block_id, pred in predictions.items():
        if block_id not in blocks:
            continue
        block = blocks[block_id]
        if block["group"].lower() in RESERVED_BENCHMARK_REPOS:
            continue

        labels = pred["labels"]
        for q, label in zip(block["questions"], labels):
            if label is None or label == -100:
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
            try:
                validate(row)
                trace_rows.append(row)
            except Exception:
                continue

    # Semantic pairs
    sem_path = ROOT / "data/teacher/semantic-pairs.jsonl"
    sem_rows = []
    if sem_path.exists():
        for line in sem_path.read_text().splitlines():
            if line.strip():
                pair = json.loads(line)
                if pair["group"].lower() in RESERVED_BENCHMARK_REPOS:
                    continue
                q_text = pair["query"].strip()
                is_pos = pair["pair_role"] == "paired_documentation"
                state = pair["state"]
                group = pair["group"]
                split = pair["split"]

                # Score row
                r_score = {
                    "schema_version": SCHEMA_VERSION,
                    "id": f"sem_{pair['id']}_score",
                    "group": group,
                    "command": "git grep -n -C 5",
                    "state": state,
                    "question": {
                        "type": "score",
                        "instructions": f"How closely does this code match the requested functionality: {q_text}?",
                        "criteria": [
                            "No relevance / shares keywords only",
                            "Partially relevant context",
                            "Direct implementation of requested functionality"
                        ]
                    },
                    "label": 2 if is_pos else 0,
                    "split": split,
                    "domain": "code_search"
                }
                # Choice row
                r_choice = {
                    "schema_version": SCHEMA_VERSION,
                    "id": f"sem_{pair['id']}_choice",
                    "group": group,
                    "command": "git grep -n -C 5",
                    "state": state,
                    "question": {
                        "type": "choice",
                        "instructions": f"Evaluate this code against the search query: {q_text}",
                        "criteria": [
                            "Direct implementation of the query",
                            "Lexical overlap but different implementation",
                            "Unrelated code"
                        ]
                    },
                    "label": 0 if is_pos else 1,
                    "split": split,
                    "domain": "code_search"
                }
                # Noul row
                r_noul = {
                    "schema_version": SCHEMA_VERSION,
                    "id": f"sem_{pair['id']}_noul",
                    "group": group,
                    "command": "git grep -n -C 5",
                    "state": state,
                    "question": {
                        "type": "noul",
                        "instructions": f"Does this code implement the following functionality: {q_text}?"
                    },
                    "label": 1 if is_pos else 0,
                    "split": split,
                    "domain": "code_search"
                }
                for r in (r_score, r_choice, r_noul):
                    try:
                        validate(r)
                        sem_rows.append(r)
                    except Exception:
                        continue

    print(f"  Loaded {len(trace_rows)} SWE trace rows and {len(sem_rows)} code search rows.")
    return trace_rows + sem_rows


def main():
    print("=== Assembling Scaled Multi-Task Diverse Dataset ===")

    # 1. Noul: BoolQ + SQuAD 2.0
    boolq_rows = build_boolq_rows(limit=4000)
    squad_rows = build_squad_rows(limit=3500)

    # 2. Choice: Amazon Massive (12..20 options) + AI2 ARC (4..5 options)
    massive_rows = build_massive_rows(limit=3800)
    arc_rows = build_arc_rows(limit=3200)

    # 3. Score: HelpSteer2 (5-level ordinal)
    helpsteer_rows = build_helpsteer_rows(limit=5000)

    # 4. Code / Agent Traces (SWE-bench + CodeSearchNet)
    code_agent_rows = load_swe_and_code_search_rows()

    all_rows = boolq_rows + squad_rows + massive_rows + arc_rows + helpsteer_rows + code_agent_rows
    print(f"\nTotal raw collected rows: {len(all_rows)}")

    # Strict isolation: Train vs Validation by source group
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "development"]

    train_groups = {r["group"].lower() for r in train_rows}
    val_groups = {r["group"].lower() for r in val_rows}
    leakage = train_groups & val_groups
    if leakage:
        print(f"Resolving {len(leakage)} overlapping groups by removing from validation: {leakage}")
        val_rows = [r for r in val_rows if r["group"].lower() not in leakage]
        val_groups = {r["group"].lower() for r in val_rows}
        assert len(train_groups & val_groups) == 0, "Zero source leakage violated!"

    # Exclude reserved benchmark repos
    train_rows = [r for r in train_rows if r["group"].lower() not in RESERVED_BENCHMARK_REPOS]
    val_rows = [r for r in val_rows if r["group"].lower() not in RESERVED_BENCHMARK_REPOS]

    random.seed(17)
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows))
    (data_dir / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val_rows))

    summary = {
        "train": {
            "total_rows": len(train_rows),
            "groups": len(train_groups),
            "by_type": dict(Counter(r["question"]["type"] for r in train_rows)),
            "by_domain": dict(Counter(r.get("domain", "unknown") for r in train_rows)),
        },
        "validation": {
            "total_rows": len(val_rows),
            "groups": len(val_groups),
            "by_type": dict(Counter(r["question"]["type"] for r in val_rows)),
            "by_domain": dict(Counter(r.get("domain", "unknown") for r in val_rows)),
        },
        "reserved_repos_excluded": sorted(RESERVED_BENCHMARK_REPOS)
    }

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "supervised-dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== Dataset Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {data_dir/'train.jsonl'} ({len(train_rows)} rows) and {data_dir/'validation.jsonl'} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
