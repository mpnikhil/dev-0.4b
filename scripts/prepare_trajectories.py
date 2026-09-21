"""Extract objective/command/observation records. Does NOT invent retention labels.

Future actions and final outcomes are annotation metadata only, never model input.
Splits are stable by repository, keeping repeated attempts at an issue together.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq

SOURCE = 'https://huggingface.co/datasets/nebius/SWE-agent-trajectories'


def split_for(repo):
    bucket = int(hashlib.sha256(repo.encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def records(row):
    turns = row['trajectory']
    initial = next((i for i, turn in enumerate(turns) if turn['role'] == 'user' and turn.get('text')), None)
    if initial is None:
        return
    objective = turns[initial]['text']
    instance = row['instance_id']
    repo = instance.rsplit('-', 1)[0].replace('__', '/')
    for i in range(initial+2, len(turns)):
        previous, current = turns[i-1], turns[i]
        if previous['role'] != 'ai' or current['role'] != 'user':
            continue
        action = previous.get('text') or ''
        observation = current.get('text') or ''
        commands = re.findall(r'```(?:bash|sh|shell)?\s*\n(.*?)```', action, re.S)
        if not commands or not observation.strip():
            continue
        command = commands[-1].strip()
        digest = hashlib.sha256((instance+'\0'+command+'\0'+observation).encode()).hexdigest()
        yield {'id': digest, 'group': repo, 'instance_id': instance, 'source': SOURCE,
               'objective': objective, 'command': command, 'prior_agent_message': action,
               'state': observation, 'label_status': 'unlabeled',
               'annotation_only': {'next_agent_message': turns[i+1].get('text') if i+1 < len(turns) else None,
                                   'trajectory_succeeded': bool(row['target']),
                                   'generator': row['model_name'], 'turn_index': i}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', default='.cache/hf-data/swe-agent/data/train-00000-of-00012.parquet')
    p.add_argument('--output', default='data/hf')
    p.add_argument('--per-repo', type=int, default=100)
    a = p.parse_args()
    if a.per_repo <= 0:
        p.error('--per-repo must be positive')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    seen, repositories, splits, kinds = set(), Counter(), Counter(), Counter()
    paths = {split: out/f'{split}.unlabeled.jsonl' for split in ['train','validation','test']}
    files = {split: path.open('w') for split, path in paths.items()}
    trajectories = 0
    try:
        for batch in pq.ParquetFile(a.parquet).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                trajectories += 1
                for record in records(row):
                    repo = record['group']
                    if record['id'] in seen or repositories[repo] >= a.per_repo:
                        continue
                    seen.add(record['id']); repositories[repo] += 1
                    split = split_for(repo); splits[split] += 1
                    command = record['command']
                    kind = ('search' if re.search(r'\b(rg|grep|search_dir|search_file|find_file)\b', command)
                            else 'test' if re.search(r'\b(pytest|tox|unittest|test)\b', command)
                            else 'other')
                    kinds[kind] += 1
                    files[split].write(json.dumps(record)+'\n')
    finally:
        for f in files.values():
            f.close()
    source = Path(a.parquet)
    summary = {'source': SOURCE, 'source_file': source.name,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'dataset_license': 'CC-BY-4.0; see source card for upstream repository and model-output conditions',
               'trajectories_scanned': trajectories, 'deduplicated_observations': len(seen),
               'repositories': len(repositories), 'splits': dict(splits), 'command_kinds': dict(kinds),
               'split_unit': 'repository', 'labels': 'NONE: annotation required',
               'sampling': f'First {a.per_repo} deduplicated observations per repository in first shard; not representative'}
    (out/'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
