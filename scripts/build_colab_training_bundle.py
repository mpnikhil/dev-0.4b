"""Bundle project source, dataset, and remote training runner for Colab A100."""
import base64
import io
import json
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="answerdotai/ModernBERT-large", help="Hugging Face model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--accumulation", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    train_file = ROOT / "data/train.jsonl"
    val_file = ROOT / "data/validation.jsonl"
    if not train_file.exists() or not val_file.exists():
        print("Warning: data/train.jsonl or data/validation.jsonl not found yet. Bundle will require building dataset first.")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        # Add src/dev
        for py_file in (ROOT / "src/dev").glob("*.py"):
            tar.add(py_file, arcname=f"src/dev/{py_file.name}")
        # Add datasets if present
        if train_file.exists():
            tar.add(train_file, arcname="data/train.jsonl")
        if val_file.exists():
            tar.add(val_file, arcname="data/validation.jsonl")
        # Add benchmark data and script
        csn_bench = ROOT / "data/benchmarks/csn-human-python.jsonl"
        if csn_bench.exists():
            tar.add(csn_bench, arcname="data/benchmarks/csn-human-python.jsonl")
        eval_script = ROOT / "scripts/evaluate_csn_benchmark.py"
        if eval_script.exists():
            tar.add(eval_script, arcname="scripts/evaluate_csn_benchmark.py")

    payload_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    remote_code = f'''# Auto-generated Colab A100 Training Runner
import base64, io, os, subprocess, sys, tarfile, time, json
from pathlib import Path

print("=== 1. System & GPU Check ===")
import torch
print(f"PyTorch: {{torch.__version__}}, CUDA Available: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"Device: {{torch.cuda.get_device_name()}}, VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}} GB")
    print(f"BF16 Supported: {{torch.cuda.is_bf16_supported()}}")

print("=== 2. Install Dependencies ===")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48", "safetensors", "numpy", "peft"], check=True)

print("=== 3. Extract Payload ===")
workspace = Path("/content/jev-workspace")
workspace.mkdir(parents=True, exist_ok=True)
payload_bytes = base64.b64decode({repr(payload_b64)})
with tarfile.open(fileobj=io.BytesIO(payload_bytes), mode="r:gz") as tar:
    tar.extractall(workspace)

print(f"Extracted files to {{workspace}}:")
for p in workspace.rglob("*"):
    if p.is_file():
        print(f"  {{p.relative_to(workspace)}} ({{p.stat().st_size}} bytes)")

print("=== 4. Launch Scaled 3-Head Training ({args.model}) ===")
train_script = workspace / "src/dev/train.py"
env = dict(os.environ, PYTHONPATH=str(workspace / "src"), HF_HUB_DISABLE_PROGRESS_BARS="1")
cmd = [
    sys.executable, "-m", "dev.train",
    "--model", "{args.model}",
    "--train", str(workspace / "data/train.jsonl"),
    "--validation", str(workspace / "data/validation.jsonl"),
    "--output", str(workspace / "runs/jev-model-a100"),
    "--device", "cuda",
    "--epochs", "{args.epochs}",
    "--batch-size", "{args.batch_size}",
    "--accumulation", "{args.accumulation}",
    "--max-length", "1024",
    "--encoder-lr", "1.5e-5",
    "--head-lr", "3e-4"
]
start_t = time.time()
res = subprocess.run(cmd, env=env, cwd=str(workspace))
print(f"Training exit code: {{res.returncode}}, elapsed: {{time.time()-start_t:.1f}}s")
if res.returncode != 0:
    print("ERROR: Training failed!")
    sys.exit(res.returncode)

print("=== 5. Run Benchmark Evaluation on GPU ===")
bench_cmd = [
    sys.executable, str(workspace / "scripts/evaluate_csn_benchmark.py"),
    "--checkpoint", str(workspace / "runs/jev-model-a100"),
    "--benchmark", str(workspace / "data/benchmarks/csn-human-python.jsonl"),
    "--device", "cuda",
    "--max-length", "4096",
    "--output", str(workspace / "runs/jev-model-a100/csn-benchmark-evaluation.json")
]
eval_res = subprocess.run(bench_cmd, env=env, cwd=str(workspace))
print(f"Benchmark eval exit code: {{eval_res.returncode}}")

print("=== 6. Prepare Download Chunks & Inference Archive ===")
out_dir = workspace / "runs/jev-model-a100"
chunks_dir = Path("/content/chunks")
chunks_dir.mkdir(parents=True, exist_ok=True)
subprocess.run(["split", "-b", "20M", "-d", str(out_dir / "model.safetensors"), "/content/chunks/part_"], check=True)
print(f"Created {{len(list(chunks_dir.glob('part_*')))}} chunks in /content/chunks")

inf_archive = Path("/content/jev-model-inference.tar.gz")
with tarfile.open(inf_archive, "w:gz") as tar:
    for item in out_dir.glob("*"):
        if item.name != "trainer.pt":
            tar.add(item, arcname=f"jev-model-a100/{{item.name}}")
print(f"Inference archive created: {{inf_archive}} ({{inf_archive.stat().st_size}} bytes)")
'''

    out_script = ROOT / ".cache/train_colab_remote.py"
    out_script.parent.mkdir(exist_ok=True)
    out_script.write_text(remote_code)
    print(f"Wrote remote Colab script: {out_script} ({out_script.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
