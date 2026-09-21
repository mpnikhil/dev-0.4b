"""Version 2: caller-defined questions and criteria; one question per training row."""
import json
from pathlib import Path
import torch

SCHEMA_VERSION = 2
HEADS = ('choice', 'score', 'noul')


def validate(row):
    if row.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Require schema_version=2; fixed-task v1 data must be explicitly migrated')
    for key in ('group', 'state'):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f'{key} must be a nonempty string')
    if not isinstance(row.get('command', ''), str):
        raise ValueError('command must be a string')
    q = row.get('question', {})
    if q.get('type') not in HEADS or not isinstance(q.get('instructions'), str) or not q['instructions'].strip():
        raise ValueError('Require question type and nonempty instructions')
    kind = q['type']
    criteria = q.get('criteria', [])
    if kind == 'noul':
        if criteria and [c.strip().lower() for c in criteria] not in (['no', 'yes'], ['false', 'true']):
            raise ValueError('Noul criteria, if specified, must be ["No", "Yes"]')
        size = 2
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= (10 if kind == 'score' else 255):
            raise ValueError('Choice requires 2..255 criteria; Score requires 2..10 ordered levels')
        if any(not isinstance(s, str) or not s.strip() for s in criteria) or len(set(criteria)) != len(criteria):
            raise ValueError('Criteria must be unique nonempty descriptions')
        size = len(criteria)
    label = row.get('label', -100)
    if type(label) is not int or (label != -100 and not 0 <= label < size):
        raise ValueError(f'Invalid {kind} label: {label}')
    return row


def read_rows(path):
    rows = [validate(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def encode(row, tokenizer, max_length):
    validate(row)
    q = row['question']
    text = f"QUESTION TYPE: {q['type']}\nQUESTION: {q['instructions']}\nCOMMAND: {row.get('command', '')}\nSTATE:\n{row['state']}\nCRITERIA:\n"
    spans = []
    criteria = q.get('criteria') or (['No', 'Yes'] if q['type'] == 'noul' else [])
    for i, criterion in enumerate(criteria):
        text += f'{i}: '
        start = len(text)
        text += criterion
        spans.append((start, len(text)))
        text += '\n'
    text += 'ANSWER:\n'
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=False)
    if len(encoded['input_ids']) > max_length:
        raise ValueError(f"Input has {len(encoded['input_ids'])} tokens, limit {max_length}; chunk it, do not truncate")
    offsets = encoded.pop('offset_mapping')
    masks = [[int(end > a and start < b and end > start) for start, end in offsets] for a, b in spans]
    if any(not any(mask) for mask in masks):
        raise ValueError('Criterion lost during tokenization')
    return {'input_ids': encoded['input_ids'], 'candidate_mask': masks,
            'labels': {k: row.get('label', -100) if k == q['type'] else -100 for k in HEADS}}


def collate(rows, pad_id):
    batch, length, candidates = len(rows), max(len(r['input_ids']) for r in rows), max(len(r['candidate_mask']) for r in rows)
    result = {'input_ids': torch.full((batch, length), pad_id, dtype=torch.long),
              'attention_mask': torch.zeros(batch, length, dtype=torch.long),
              'candidate_mask': torch.zeros(batch, candidates, length),
              'candidate_valid': torch.zeros(batch, candidates, dtype=torch.bool)}
    for i, row in enumerate(rows):
        n, c = len(row['input_ids']), len(row['candidate_mask'])
        result['input_ids'][i, :n] = torch.tensor(row['input_ids'])
        result['attention_mask'][i, :n] = 1
        result['candidate_mask'][i, :c, :n] = torch.tensor(row['candidate_mask'])
        result['candidate_valid'][i, :c] = True
    result['labels'] = {k: torch.tensor([r['labels'][k] for r in rows]) for k in HEADS}
    return result
