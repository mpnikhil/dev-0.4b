"""Reproducible audit of available raw material; never counts raw data as gold labels."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def command_kind(command):
    if re.search(r'\b(rg|grep|search_dir|search_file|find_file)\b', command):
        return 'search'
    if re.search(r'\b(pytest|unittest|tox|nosetests)\b|\b(npm|cargo|go)\s+test', command):
        return 'test'
    if re.search(r'\b(make|cmake|gcc|clang|tsc)\b|\b(npm|cargo|go)\s+build', command):
        return 'build'
    return 'other'


def percentiles(values):
    return {str(q): int(np.percentile(values, q)) for q in [50, 90, 95, 99]} if values else {}


def main():
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base', local_files_only=True,
                                             model_max_length=10**9)
    rows = []
    for p in sorted((ROOT/'data/hf').glob('*.unlabeled.jsonl')):
        rows.extend(json.loads(s) for s in p.read_text().splitlines())
    stats, counts, repo_counts = defaultdict(list), Counter(), defaultdict(set)
    token_lengths = []
    for start in range(0, len(rows), 128):
        texts = [r['state'] for r in rows[start:start+128]]
        token_lengths.extend(len(x) for x in tokenizer(texts, add_special_tokens=False)['input_ids'])
    raw_hashes = Counter()
    heavy_candidates = []
    for row, length in zip(rows, token_lengths):
        kind = command_kind(row['command'])
        stats[kind].append(length); counts[kind] += 1; repo_counts[kind].add(row['group'])
        if re.search(r'\brg\b', row['command']):
            counts['literal_rg_commands'] += 1
        raw_hashes[hashlib.sha256(row['state'].encode()).hexdigest()] += 1
        if kind != 'other' and length >= 1024:
            heavy_candidates.append({'id':row['id'],'repo':row['group'],'instance_id':row['instance_id'],
                                     'kind':kind,'tokens':length,'command':row['command']})
    verified = ROOT/'.cache/hf-data/swebench-verified/data/test-00000-of-00001.parquet'
    overlap = {}
    if verified.exists():
        v = pq.read_table(verified, columns=['instance_id','repo']).to_pylist()
        vr = {r['repo'] for r in v}; vi = {r['instance_id'] for r in v}
        overlap = {'verified_tasks':len(v),'verified_repos':len(vr),
                   'overlapping_repos': sorted(vr & {r['group'] for r in rows}),
                   'overlapping_instance_ids': sorted(vi & {r['instance_id'] for r in rows}),
                   'observations_from_verified_repos': sum(r['group'] in vr for r in rows)}
    report = {'source_manifest':json.loads((ROOT/'data/hf/manifest.json').read_text()),
              'tokenizer':'answerdotai/ModernBERT-base; counts are not Claude/Codex tokens',
              'total_observations':len(rows),'unique_output_texts':len(raw_hashes),
              'commands':dict(counts),'by_kind':{},'verified_overlap':overlap,
              'labels_available_for_our_task':0}
    for kind, lengths in stats.items():
        report['by_kind'][kind] = {'observations':len(lengths),'repositories':len(repo_counts[kind]),
                                   'token_percentiles':percentiles(lengths),
                                   'at_least_1024_tokens':sum(x>=1024 for x in lengths),
                                   'at_least_4096_tokens':sum(x>=4096 for x in lengths),
                                   'at_least_8192_tokens':sum(x>=8192 for x in lengths)}
    out=ROOT/'docs/data-audit.json';out.write_text(json.dumps(report,indent=2))
    (ROOT/'.cache/audit/heavy-candidates.json').write_text(json.dumps(heavy_candidates,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
