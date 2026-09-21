"""Prepare trace blocks for weak teacher supervision; preserve exact source offsets."""
import gzip
import hashlib
import json
import re
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base', local_files_only=True)
    repoqa = json.loads(gzip.decompress((ROOT/'.cache/benchmarks/repoqa.json.gz').read_bytes()))
    reserved = {r['repo'].lower() for rows in repoqa.values() for r in rows}
    out = ROOT/'data/teacher'; out.mkdir(parents=True, exist_ok=True)
    seen = set(); counts = {}; exclusions = Counter()
    # Dev first: remove matching normalized train blocks, keeping validation intact.
    for split in ['development', 'train']:
        grouped = {}
        for line in (ROOT/f'data/pilot/{split}.annotation.jsonl').read_text().splitlines():
            row = json.loads(line)
            grouped.setdefault(row['source']['id'], []).append(row)
        records = []
        for source_id, rows in grouped.items():
            row = rows[0]
            if row['group'].lower() in reserved:
                exclusions['repoqa_repository'] += 1; continue
            state = row['state']
            enc = tokenizer(state, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc['offset_mapping']
            # Select a reproducible block anywhere in the output, before labeling.
            width = 700
            block = int(source_id[:8], 16) % max(1, (len(offsets)+width-1)//width)
            lo, hi = block*width, min((block+1)*width, len(offsets))
            start, end = offsets[lo][0], offsets[hi-1][1]
            chunk = state[start:end]
            fingerprint = hashlib.sha256(re.sub(r'\s+', ' ', chunk).strip().encode()).hexdigest()
            if fingerprint in seen:
                exclusions['normalized_duplicate_block'] += 1; continue
            seen.add(fingerprint)
            questions = [r['question'] for r in rows]
            records.append({'id': source_id, 'split': split, 'group': row['group'], 'kind': row['source']['kind'],
                            'command': row['command'], 'state': chunk, 'questions': questions,
                            'source': row['source'], 'source_byte_span': [len(state[:start].encode()), len(state[:end].encode())],
                            'scope': 'supplied_excerpt_only', 'complete_original': start == 0 and end == len(state)})
        (out/f'{split}.blocks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        counts[split] = {'blocks': len(records), 'kinds': dict(Counter(r['kind'] for r in records))}
    report = {'counts': counts, 'exclusions': dict(exclusions), 'max_state_tokens': 700,
              'scope': 'block-level weak supervision; no whole-log absence labels',
              'teacher': 'mlx-community/Qwen3-4B-Instruct-2507-4bit',
              'revision': '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'}
    (ROOT/'docs/teacher-data.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
