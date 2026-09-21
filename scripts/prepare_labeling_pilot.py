"""Freeze repo splits and a 600-output annotation queue. Does not create gold labels."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def partition(repo):
    value = int(hashlib.sha256(('jev-v1:' + repo).encode()).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else 'development' if value < 85 else 'calibration' if value < 92 else 'test'


def sample(rows, count):
    """Round robin across repositories to limit domination by repeated issue attempts."""
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['id']):
        buckets[row['repo']].append(row)
    selected = []
    while len(selected) < count and any(buckets.values()):
        for repo in sorted(buckets):
            if buckets[repo]:
                selected.append(buckets[repo].pop(0))
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} eligible records, need {count}')
    return selected


def questions(kind):
    if kind == 'test':
        return [
            {'type': 'choice', 'instructions': 'Which test-run events are explicitly reported in this output?',
             'criteria': ['Assertion failures only', 'Collection or setup errors only', 'Both assertion failures and collection or setup errors', 'Neither is established by this output']},
            {'type': 'noul', 'instructions': 'Does the output explicitly report at least one failed assertion in a test?'},
            {'type': 'score', 'instructions': 'How much direct evidence of an assertion failure does this output provide?',
             'criteria': ['No assertion failure is reported', 'An assertion failure is reported without the failing test or assertion details', 'An assertion failure is tied to a specific test but assertion details are absent', 'An assertion failure, its test identity, and assertion details are shown']},
        ]
    return [
        {'type': 'choice', 'instructions': 'What kind of matches does this code-search output show?',
         'criteria': ['Executable implementation code only', 'Tests or examples only', 'Documentation or comments only', 'A mixture of these kinds', 'Insufficient context to determine']},
        {'type': 'noul', 'instructions': 'Does the output show executable implementation code, beyond merely documenting or testing the searched behavior?'},
        {'type': 'score', 'instructions': 'How directly does this output reveal an implementation relevant to the search command?',
         'criteria': ['No relevant implementation evidence', 'Names or references without implementation details', 'Some relevant implementation details but necessary context is missing', 'A relevant implementation is shown with enough local context to interpret it']},
    ]


def main():
    inventory = [json.loads(s) for s in (ROOT / '.cache/audit/candidate-inventory.jsonl').read_text().splitlines()]
    raw = {}
    for name in {r['sample'] for r in inventory}:
        for line in (ROOT / f'.cache/audit/{name}-heavy.jsonl').read_text().splitlines():
            r = json.loads(line); raw[r['id']] = r
    out = ROOT / 'data/pilot'; out.mkdir(parents=True, exist_ok=True)
    assignments = {r['repo']: partition(r['repo']) for r in inventory}
    (out / 'repository-splits.json').write_text(json.dumps(assignments, indent=2) + '\n')
    report = {'schema_version': 2, 'status': 'unlabeled_annotation_queue', 'split_rule': 'sha256(jev-v1:repo) modulo 100; 70/15/7/8',
              'reserved_benchmark_repos': json.loads((ROOT/'docs/candidate-audit.json').read_text())['reserved_repos'],
              'pool_counts': dict(Counter(assignments[r['repo']] for r in inventory)), 'selected': {},
              'limitations': ['Exact-output deduplication complete; near-duplicate audit still required before training.',
                             'Question templates use command/output kind, not future outcomes; kind classification needs review.',
                             'This queue is not the sealed benchmark, and its labels are all missing.']}
    manifest = []
    for split, count in [('train', 500), ('development', 100)]:
        candidates = [r for r in inventory if assignments[r['repo']] == split and r['kind'] in ['test', 'search']]
        chosen = sample(candidates, count)
        queue = []
        for meta in chosen:
            source = raw[meta['id']]
            for i, question in enumerate(questions(meta['kind'])):
                if question['type'] == 'choice':
                    random.Random(meta['id']).shuffle(question['criteria'])
                queue.append({'schema_version': 2, 'id': f"{meta['id']}:{i}", 'group': meta['repo'],
                              'state': source['state'], 'command': source['command'], 'question': question,
                              'label': -100, 'answerability': None, 'evidence_byte_spans': [],
                              'label_source': 'pending_review', 'source': meta,
                              'review_note': 'Use null answerability until reviewed; unknown is not a negative label. Full outputs require evidence-preserving chunk preparation before training.'})
            manifest.append({**meta, 'split': split})
        (out / f'{split}.annotation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in queue))
        report['selected'][split] = {'outputs': len(chosen), 'questions': len(queue),
                                    'repositories': len({r['repo'] for r in chosen}), 'kinds': dict(Counter(r['kind'] for r in chosen))}
    (out/'selected-manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (ROOT/'docs/labeling-pilot.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
