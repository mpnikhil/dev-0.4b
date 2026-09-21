"""Freeze independent human evaluation and prepare teacher-rated semantic pairs."""
import ast
import csv
import gzip
import hashlib
import json
import random
import re
import textwrap
from collections import defaultdict, Counter
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from dev.retrieval import BM25

ROOT = Path(__file__).resolve().parents[1]

def h(text): return hashlib.sha256(text.encode()).hexdigest()

def strip_docstring(code):
    code = textwrap.dedent(code)
    try:
        tree = ast.parse(code)
        node = tree.body[0]
        if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            doc = node.body[0]
            lines=code.splitlines(keepends=True)
            return ''.join(lines[:doc.lineno-1]+lines[doc.end_lineno:])
    except (SyntaxError,IndexError): pass
    return None  # exclude rather than leak the query through an unparsed docstring


def main():
    out=ROOT/'data/benchmarks';out.mkdir(parents=True,exist_ok=True)
    root=ROOT/'.cache/benchmarks'
    annotations=[r for r in csv.DictReader((root/'codesearchnet-annotations.csv').open()) if r['Language'].lower()=='python']
    snippets={r['url']:r for r in map(json.loads,(root/'csn-python-snippets.jsonl').read_text().splitlines())}
    missing={u for u,r in snippets.items() if 'code' not in r}
    # Recover deleted upstream sources only by exact corpus URL, never by a similar function.
    train_file=ROOT/'.cache/hf-data/codesearchnet/python/train-00000-of-00001.parquet'
    for batch in pq.ParquetFile(train_file).iter_batches(batch_size=4096,columns=['func_code_url','func_code_string','repository_name']):
        for r in batch.to_pylist():
            if r['func_code_url'] in missing:
                snippets[r['func_code_url']]={'url':r['func_code_url'],'repo':r['repository_name'],'code':r['func_code_string'],'source':'HF train exact URL recovery'}
                missing.remove(r['func_code_url'])
    relevance=defaultdict(lambda:defaultdict(list))
    for r in annotations: relevance[r['Query']][r['GitHubUrl']].append(float(r['Relevance']))
    complete={q:rs for q,rs in relevance.items() if all(u not in missing for u in rs)}
    ordered=sorted(complete,key=lambda q:h('csn-human-v1:'+q))
    split_at=round(len(ordered)*0.7)
    assignments={q:('development' if i<split_at else 'test') for i,q in enumerate(ordered)}
    benchmark=[]
    for q,rs in complete.items():
        benchmark.append({'id':h(q),'query':q,'split':assignments[q],
                          'candidates':[{'url':u,'repo':snippets[u]['repo'],'code':snippets[u]['code'],'relevance':sum(v)/len(v),'annotation_count':len(v)} for u,v in sorted(rs.items())]})
    (out/'csn-human-python.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in benchmark))
    repoqa=json.loads(gzip.decompress((root/'repoqa.json.gz').read_bytes()))
    reserved={r['repo'].lower() for rows in repoqa.values() for r in rows}
    reserved.update('/'.join(r['GitHubUrl'].split('/')[3:5]).lower() for r in annotations)
    reserved.update(r.lower() for r in json.loads((ROOT/'docs/candidate-audit.json').read_text())['reserved_repos'])
    tokenizer=AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base',local_files_only=True)
    pool=[];seen=set()
    # Fixed hash sample of corpus; exclude all human-benchmark and RepoQA repositories.
    for batch in pq.ParquetFile(train_file).iter_batches(batch_size=2048):
        for r in batch.to_pylist():
            if r['repository_name'].lower() in reserved: continue
            if int(h(r['func_code_url'])[:8],16)%80: continue
            query=r['func_documentation_string'].strip().split('\n')[0].strip()
            if not 30<=len(query)<=240 or query.lower() in {q.lower() for q in relevance}: continue
            code=strip_docstring(r['func_code_string'])
            if not code or not 50<=len(code)<=4000:continue
            if len(tokenizer.encode(code+query,add_special_tokens=False))>850:continue
            digest=h(re.sub(r'\s+',' ',code).strip())
            if digest in seen:continue
            seen.add(digest)
            pool.append({'query':query,'code':code,'repo':r['repository_name'],'url':r['func_code_url']})
    if len(pool)<600:raise ValueError(f'Only {len(pool)} eligible functions')
    random.Random(17).shuffle(pool)
    pairdir=ROOT/'data/teacher';pairdir.mkdir(exist_ok=True)
    records=[];counts={}
    for split,limit in [('development',50),('train',250)]:
        candidates=[r for r in pool if ('development' if int(h(r['repo'].lower())[:8],16)%10==0 else 'train')==split]
        index=BM25([r['code'] for r in candidates])
        for pos,row in enumerate(candidates[:limit]):
            scores=index.scores(row['query'])
            hard=next(i for i in sorted(range(len(scores)),key=lambda i:-scores[i]) if i!=pos and candidates[i]['repo']!=row['repo'])
            for role,other in [('paired_documentation',row),('lexically_hard_candidate',candidates[hard])]:
                records.append({'id':h(row['url']+'\0'+other['url']),'split':split,'query':row['query'],'state':other['code'],
                                'group':row['repo'],'candidate_repo':other['repo'],'source_url':other['url'],
                                'query_source_url':row['url'],'pair_role':role,'labels':None,
                                'scope':'supplied_code_only','transformation':'function docstring removed; query first documentation line'})
        counts[split]=sum(r['split']==split for r in records)
    (pairdir/'semantic-pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    manifest={'benchmark':'CodeSearchNet human judgments, Python judged-pool reranking adaptation',
              'annotation_sha256':h((root/'codesearchnet-annotations.csv').read_text()),
              'all_queries':len(relevance),'complete_queries':len(complete),'missing_source_urls':len(missing),
              'excluded_incomplete_queries':sorted(set(relevance)-set(complete)),
              'split_queries':dict(Counter(assignments.values())),'candidate_pairs':sum(len(r['candidates']) for r in benchmark),
              'source_note':'Only fully reconstructed query pools included. Not a full-corpus CodeSearchNet leaderboard score.',
              'teacher_pair_counts':counts,'candidate_function_pool':len(pool),'reserved_repositories':len(reserved),
              'training_corpus_revision':'bd0cf261e357a3eb5c8fba490d23ec1a1cd59555',
              'repoqa_release_sha256':hashlib.sha256((root/'repoqa.json.gz').read_bytes()).hexdigest(),
              'repoqa_release_tasks':sum(len(r['needles']) for rows in repoqa.values() for r in rows)}
    (ROOT/'docs/semantic-experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
