"""Inventory unlabeled large-output candidates; exclude downstream benchmark repos."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'swesmith_tool_shard0': ('SWE-bench/SWE-smith-trajectories', '08e109b4a59eaeebf80e4675cd125d42e7ac99a4'),
    'openhands_first128': ('nebius/SWE-rebench-openhands-trajectories', '35455389ab51bf5e2306bfd436ef72d0f98bf882'),
}


def main():
    verified = pq.read_table(
        ROOT / '.cache/hf-data/swebench-verified/data/test-00000-of-00001.parquet',
        columns=['repo'],
    ).to_pylist()
    reserved = {row['repo'] for row in verified}
    seen = set()
    inventory = []
    excluded = 0
    total = 0
    for name, (dataset, revision) in SOURCES.items():
        for line in (ROOT / f'.cache/audit/{name}-heavy.jsonl').read_text().splitlines():
            row = json.loads(line)
            total += 1
            if row['repo'] in reserved:
                excluded += 1
                continue
            output_hash = hashlib.sha256(row['state'].encode()).hexdigest()
            if output_hash in seen:
                continue
            seen.add(output_hash)
            inventory.append({
                **{k: row[k] for k in ['id', 'repo', 'instance_id', 'kind', 'tokens']},
                'dataset': dataset, 'revision': revision, 'sample': name,
                'output_sha256': output_hash, 'label_status': 'unlabeled',
            })
    report = {
        'combined_candidates': total,
        'excluded_verified_repo_candidates': excluded,
        'exact_output_deduplicated_candidates': len(inventory),
        'repositories': len({r['repo'] for r in inventory}),
        'kinds': dict(Counter(r['kind'] for r in inventory)),
        'reserved_repos': sorted(reserved),
        'limitations': [
            'Nonrandom source samples; command kinds are regex heuristics.',
            'At least 1024 ModernBERT tokens, not target-agent tokens.',
            'Exact output deduplication only; semantic duplicates remain possible.',
            'Neither gold labels nor a finalized benchmark split.',
            'Only two audited source samples; no extrapolation to full corpora.',
        ],
    }
    (ROOT / 'docs/candidate-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / '.cache/audit/candidate-inventory.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in inventory)
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
