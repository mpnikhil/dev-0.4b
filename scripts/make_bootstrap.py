"""Synthetic schema-v2 plumbing fixture, never a coding-utility benchmark."""
import argparse
import json
import random
from pathlib import Path


def generate(groups, split, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(groups):
        assertion = i % 2 == 0
        state = (f'FAILED tests/test_cache_{i}.py::test_timeout\nAssertionError: expected 200, got 500\n1 failed'
                 if assertion else f'ERROR collecting tests/test_cache_{i}.py\nModuleNotFoundError: No module named widget_{i}\nInterrupted: 1 error during collection')
        options = ['A test assertion failed', 'Tests could not be collected', 'The output does not establish either']
        rng.shuffle(options)
        correct = 'A test assertion failed' if assertion else 'Tests could not be collected'
        questions = [
            ({'type': 'choice', 'instructions': 'Which event does this output report?', 'criteria': options}, options.index(correct)),
            ({'type': 'noul', 'instructions': 'Does the output report a failed assertion in a test?'}, int(assertion)),
            ({'type': 'noul', 'instructions': 'Does the output report an error while collecting tests?'}, int(not assertion)),
            ({'type': 'score', 'instructions': 'How directly does this support investigating an assertion failure?',
              'criteria': ['No assertion failure is reported', 'An assertion failure is mentioned without its details', 'An assertion failure and its details are shown']}, 2 if assertion else 0),
        ]
        for question, label in questions:
            rows.append({'schema_version': 2, 'group': f'synthetic-{split}-{i}', 'state': state,
                         'command': 'pytest -q', 'question': question, 'label': label,
                         'label_source': 'synthetic_fixture_not_quality_evidence'})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=int, default=64)
    p.add_argument('--validation-groups', type=int, default=16)
    p.add_argument('--output', default='data')
    a = p.parse_args()
    if min(a.groups, a.validation_groups) < 1:
        p.error('group counts must be positive')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for split, count, seed in [('train', a.groups, 17), ('validation', a.validation_groups, 91)]:
        rows = generate(count, split, seed)
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        print(f'{split}: {len(rows)} synthetic examples')


if __name__ == '__main__':
    main()
