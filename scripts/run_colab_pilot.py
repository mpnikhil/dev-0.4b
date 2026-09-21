"""Execute a bounded pilot on an attached runtime; recover report and ALWAYS stop."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--session', default='jev-pilot')
p.add_argument('--output', default='runs/colab-pilot')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
cli = str(root/'.colab-venv/bin/colab')
base = [cli, '--config', str(root/'.cache/colab-sessions.json')]
out = root/a.output; out.mkdir(parents=True, exist_ok=True)
code = 1
try:
    subprocess.run([sys.executable, str(root/'scripts/build_pilot.py')], check=True)
    with (out/'execution.log').open('w') as log:
        result = subprocess.run(base+['exec','-s',a.session,'--timeout','900','-f',str(root/'.cache/pilot_remote.py')],
                                stdout=log, stderr=subprocess.STDOUT, timeout=1000)
        code = result.returncode
    recovered = subprocess.run(base+['download','-s',a.session,'/content/jev-pilot/report.json',str(out/'report.json')], timeout=60)
    subprocess.run(base+['download','-s',a.session,'/content/jev-pilot/process.json',str(out/'process.json')], timeout=60)
    if recovered.returncode:
        code = 1
    else:
        import json
        report = json.loads((out/'report.json').read_text())
        if len(report.get('runs', [])) != 2:
            code = 1
        if (out/'process.json').exists() and json.loads((out/'process.json').read_text())['returncode']:
            code = 1
finally:
    subprocess.run(base+['stop','-s',a.session], check=True, timeout=60)
raise SystemExit(code)
