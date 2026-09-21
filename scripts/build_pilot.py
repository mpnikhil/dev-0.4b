"""Bundle only the model source and pilot, never local credentials or caches."""
import base64
import io
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
    for file in ['src/dev/__init__.py', 'src/dev/model.py', 'scripts/pilot_gpu.py']:
        z.write(root/file, file)
payload = base64.b64encode(buffer.getvalue()).decode()
code = '''import base64, io, zipfile, sys, subprocess
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.6', 'peft>=0.14,<1', 'safetensors'], check=True, timeout=180)
folder=Path('/content/jev-pilot-code'); folder.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z: z.extractall(folder)
import os
env=dict(os.environ, PYTHONPATH=str(folder/'src'), HF_HUB_DISABLE_PROGRESS_BARS='1', PILOT_QWEN_ONLY='1')
result=subprocess.run([sys.executable, str(folder/'scripts/pilot_gpu.py')], env=env, capture_output=True, text=True, timeout=600)
print(result.stdout, flush=True)
print(result.stderr, flush=True)
folder=Path('/content/jev-pilot'); folder.mkdir(exist_ok=True)
(folder/'process.json').write_text(__import__('json').dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}))
result.check_returncode()
'''.replace('PAYLOAD', repr(payload))
out = root/'.cache/pilot_remote.py'
out.parent.mkdir(exist_ok=True)
out.write_text(code)
print(out)
