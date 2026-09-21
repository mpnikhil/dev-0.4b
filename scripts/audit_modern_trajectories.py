"""Audit actual tool-call observations, preserving source and format distinctions."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from audit_data import command_kind, percentiles

ROOT = Path(__file__).resolve().parents[1]


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(x.get('text','') for x in value if isinstance(x,dict))
    return ''


def tool_records(messages):
    calls = {}
    for message in messages:
        if message['role'] == 'assistant':
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                args = function.get('arguments') or {}
                if isinstance(args,str):
                    try: args=json.loads(args)
                    except json.JSONDecodeError: continue
                if function.get('name') in ['bash','execute_bash']:
                    calls[call['id']] = args.get('command','')
        if message['role'] == 'tool':
            ids = message.get('tool_call_ids') or [message.get('tool_call_id')]
            if len(ids) != 1 or ids[0] not in calls:
                continue  # never guess pairings when multiple tool results are merged
            yield calls[ids[0]], text_content(message.get('content'))


def summarize(rows, tokenizer, verified_repos):
    lengths = defaultdict(list); types = Counter(); repositories=set(); seen=set()
    candidates=[]; observations=[]; instances=set(); duplicate_records=0
    for repo, instance, messages in rows:
        repositories.add(repo); instances.add(instance)
        for command, output in tool_records(messages):
            if not command or not output: continue
            key=hashlib.sha256((instance+'\0'+command+'\0'+output).encode()).hexdigest()
            if key in seen:
                duplicate_records+=1;continue
            seen.add(key);observations.append((repo,instance,command,output,key))
    for start in range(0,len(observations),128):
        batch=observations[start:start+128]
        encoded=tokenizer([r[3] for r in batch],add_special_tokens=False)['input_ids']
        for (repo,instance,command,output,key),ids in zip(batch,encoded):
            n=len(ids);kind=command_kind(command);lengths[kind].append(n)
            types['literal_rg_commands'] += bool(re.search(r'\brg\b',command))
            types['has_truncation_marker'] += bool(re.search(r'truncat|output.{0,30}omitted|exceeds.{0,30}limit',output,re.I))
            if kind in ['test','build','search'] and n>=1024:
                candidates.append({'id':key,'repo':repo,'instance_id':instance,'kind':kind,
                                   'command':command,'state':output,'tokens':n})
    return {'trajectory_records':len(rows),'unique_instances':len(instances),'repositories':len(repositories),
            'deduplicated_bash_observations':len(observations),'duplicates_removed':duplicate_records,
            'checks':dict(types),'overlap_with_verified_repos':sorted(repositories & verified_repos),
            'by_kind':{k:{'count':len(v),'tokens':percentiles(v),'at_least_1024':sum(x>=1024 for x in v),
                           'at_least_4096':sum(x>=4096 for x in v)} for k,v in lengths.items()},
            'heavy_in_scope_observations':len(candidates),'supervised_labels_for_our_task':0},candidates


def main():
    tokenizer=AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base',local_files_only=True,model_max_length=10**9)
    verified=pq.read_table(ROOT/'.cache/hf-data/swebench-verified/data/test-00000-of-00001.parquet',columns=['repo']).to_pylist()
    verified_repos={r['repo'] for r in verified}
    smith=[]
    p=pq.ParquetFile(ROOT/'.cache/hf-data/swesmith/data/tool-00000-of-00008.parquet')
    for batch in p.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            messages=json.loads(row['messages'])
            smith.append((row['instance_id'].split('.')[0].replace('__','/'),row['instance_id'],messages))
    hands=[]
    for line in (ROOT/'.cache/audit/openhands-128.jsonl').read_text().splitlines():
        row=json.loads(line)
        hands.append((row['repo'],row['instance_id'],row['trajectory']))
    report={'tokenizer':'ModernBERT-base, not a target-agent tokenizer','samples':{}}
    for name,rows in [('swesmith_tool_shard0',smith),('openhands_first128',hands)]:
        summary,candidates=summarize(rows,tokenizer,verified_repos)
        report['samples'][name]=summary
        (ROOT/f'.cache/audit/{name}-heavy.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates))
    (ROOT/'docs/modern-data-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
