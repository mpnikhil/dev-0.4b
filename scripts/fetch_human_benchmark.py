"""Reconstruct pinned Python snippets for public CodeSearchNet human judgments."""
import csv
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT/'.cache/benchmarks'; cache = root/'github-source';cache.mkdir(exist_ok=True)
    rows = [r for r in csv.DictReader((root/'codesearchnet-annotations.csv').open()) if r['Language'].lower()=='python']
    urls = sorted({r['GitHubUrl'] for r in rows})
    matched = json.loads((root/'csn-matched.json').read_text())
    def fetch(url):
        m = re.fullmatch(r'https://github.com/([^/]+/[^/]+)/blob/([0-9a-f]+)/(.+)#L(\d+)-L(\d+)',url)
        if not m: return {'url':url,'error':'unrecognized pinned source URL'}
        repo,commit,path,start,end=m.groups();start,end=int(start),int(end)
        if url in matched:
            return {'url':url,'repo':repo,'code':matched[url]['func_code_string'],'source':'HF corpus exact URL match'}
        address=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
        dest=cache/(hashlib.sha256(address.encode()).hexdigest()+'.txt')
        try:
            if dest.exists(): data=dest.read_bytes()
            else:
                request=urllib.request.Request(address,headers={'User-Agent':'dev-public-benchmark/0.1'})
                with urllib.request.urlopen(request,timeout=20) as response:data=response.read(5_000_001)
                if len(data)>5_000_000:raise ValueError('source exceeds size limit')
                dest.write_bytes(data)
            lines=data.decode('utf-8').splitlines(keepends=True)
            if start<1 or end>len(lines):raise ValueError('line span outside source')
            return {'url':url,'repo':repo,'code':''.join(lines[start-1:end]),'source':'pinned GitHub lines'}
        except Exception as e:return {'url':url,'error':str(e)}
    outputs=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(fetch,urls)):
            outputs.append(r)
            if (i+1)%100==0:print(json.dumps({'fetched':i+1,'total':len(urls)}),flush=True)
    (root/'csn-python-snippets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
    print(json.dumps({'complete':len(outputs),'available':sum('code' in r for r in outputs)}),flush=True)


if __name__=='__main__':main()
