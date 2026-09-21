"""Package payload into .cache/payload.tar.gz, upload it to Colab, and generate lightweight runner."""
import argparse
import io
from pathlib import Path
import tarfile
import time

from colab_cli.state import StateStore
from colab_cli.contents import ContentsClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="jev-a100-v3")
    args = p.parse_args()

    train_file = ROOT / "data/train_v3.jsonl"
    val_file = ROOT / "data/validation_v3.jsonl"
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("data/train_v3.jsonl or data/validation_v3.jsonl not found")

    payload_path = ROOT / ".cache/payload.tar.gz"
    payload_path.parent.mkdir(exist_ok=True)

    print("Building .cache/payload.tar.gz...")
    with tarfile.open(payload_path, mode="w:gz") as tar:
        for py_file in (ROOT / "src/dev").glob("*.py"):
            tar.add(py_file, arcname=f"src/dev/{py_file.name}")
        tar.add(train_file, arcname="data/train_v3.jsonl")
        tar.add(val_file, arcname="data/validation_v3.jsonl")

        csn_bench = ROOT / "data/benchmarks/csn-human-python.jsonl"
        if csn_bench.exists():
            tar.add(csn_bench, arcname="data/benchmarks/csn-human-python.jsonl")
        eval_script = ROOT / "scripts/evaluate_csn_benchmark.py"
        if eval_script.exists():
            tar.add(eval_script, arcname="scripts/evaluate_csn_benchmark.py")
        comm_script = ROOT / "scripts/evaluate_community_benchmarks.py"
        if comm_script.exists():
            tar.add(comm_script, arcname="scripts/evaluate_community_benchmarks.py")

        chk_dir = ROOT / "runs/jev-modernbert-large-a100"
        for item in ["run.json"]:
            if (chk_dir / item).exists():
                tar.add(chk_dir / item, arcname=f"runs/phase2-checkpoint/{item}")
        if (chk_dir / "encoder/config.json").exists():
            tar.add(chk_dir / "encoder/config.json", arcname="runs/phase2-checkpoint/encoder/config.json")
        if (chk_dir / "tokenizer").exists():
            for f in (chk_dir / "tokenizer").iterdir():
                tar.add(f, arcname=f"runs/phase2-checkpoint/tokenizer/{f.name}")

    size_mb = payload_path.stat().st_size / 1024 / 1024
    print(f"Created {payload_path} ({size_mb:.2f} MB)")

    # Upload payload.tar.gz to Colab via HTTP PUT
    print(f"Uploading payload.tar.gz to session '{args.session}'...")
    store = StateStore()
    session = store.get(args.session)
    if not session:
        raise ValueError(f"Session {args.session} not found")
    client = ContentsClient(session)

    t0 = time.time()
    client.upload(str(payload_path), "/content/payload.tar.gz")
    print(f"Uploaded /content/payload.tar.gz in {time.time()-t0:.1f}s")

    # Generate lightweight runner script (no embedded data)
    runner_code = '''# Phase 3 Colab Runner (Lightweight)
import os, subprocess, sys, tarfile, time, json
from pathlib import Path

print("=== 1. System & GPU Check ===", flush=True)
import torch
print(f"PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}", flush=True)
else:
    raise RuntimeError("No GPU detected! Terminating.")

print("=== 2. Install Dependencies ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.48", "safetensors", "numpy", "peft", "datasets"], check=True)

print("=== 3. Extract Payload ===", flush=True)
workspace = Path("/content/jev-workspace")
workspace.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/payload.tar.gz", mode="r:gz") as tar:
    tar.extractall(workspace)

print(f"Payload extracted to {workspace}", flush=True)

print("=== 4. Reassemble Phase 2 Checkpoint ===", flush=True)
chk_dir = workspace / "runs/phase2-checkpoint"
chk_dir.mkdir(parents=True, exist_ok=True)
model_weights = chk_dir / "model.safetensors"

chunks_dir = Path("/content/chunks")
parts = sorted(chunks_dir.glob("part_*"))
print(f"Reassembling {len(parts)} chunks into {model_weights}...", flush=True)
with open(model_weights, "wb") as out:
    for part in parts:
        out.write(part.read_bytes())
print(f"Reassembled checkpoint: {model_weights.stat().st_size:,} bytes", flush=True)

print("=== 5. Launch Scaled 3-Head Training ===", flush=True)
env = dict(os.environ, PYTHONPATH=str(workspace / "src"), HF_HUB_DISABLE_PROGRESS_BARS="1")
cmd = [
    sys.executable, "-m", "dev.train",
    "--model", "answerdotai/ModernBERT-large",
    "--init-checkpoint", str(chk_dir),
    "--train", str(workspace / "data/train_v3.jsonl"),
    "--validation", str(workspace / "data/validation_v3.jsonl"),
    "--output", str(workspace / "runs/jev-phase3-a100"),
    "--device", "cuda",
    "--epochs", "2",
    "--batch-size", "8",
    "--accumulation", "4",
    "--max-length", "1024",
    "--encoder-lr", "8e-6",
    "--head-lr", "1.5e-4",
    "--gradient-checkpointing",
]
t0 = time.time()
res = subprocess.run(cmd, env=env, cwd=str(workspace))
elapsed = time.time() - t0
print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min), exit code: {res.returncode}", flush=True)
if res.returncode != 0:
    sys.exit(res.returncode)

print("=== 6. Run Benchmark Evaluation ===", flush=True)
bench_data = workspace / "data/benchmarks/csn-human-python.jsonl"
if bench_data.exists():
    bench_cmd = [
        sys.executable, str(workspace / "scripts/evaluate_csn_benchmark.py"),
        "--checkpoint", str(workspace / "runs/jev-phase3-a100"),
        "--benchmark", str(bench_data),
        "--device", "cuda",
        "--max-length", "4096",
        "--output", str(workspace / "runs/jev-phase3-a100/csn-benchmark-evaluation.json")
    ]
    subprocess.run(bench_cmd, env=env, cwd=str(workspace))

comm_script = workspace / "scripts/evaluate_community_benchmarks.py"
if comm_script.exists():
    comm_cmd = [
        sys.executable, str(comm_script),
        "--checkpoint", str(workspace / "runs/jev-phase3-a100"),
        "--device", "cuda",
        "--samples", "500",
        "--output", str(workspace / "runs/jev-phase3-a100/community-benchmark-evaluation.json")
    ]
    subprocess.run(comm_cmd, env=env, cwd=str(workspace))

print("=== 7. Prepare Download Chunks ===", flush=True)
out_dir = workspace / "runs/jev-phase3-a100"
chunks_out = Path("/content/phase3-chunks")
chunks_out.mkdir(parents=True, exist_ok=True)
subprocess.run(["split", "-b", "20M", "-d", str(out_dir / "model.safetensors"), str(chunks_out / "part_")], check=True)
print(f"Created {len(list(chunks_out.glob('part_*')))} chunks in {chunks_out}", flush=True)

run_json = out_dir / "run.json"
if run_json.exists():
    print("=== FINAL RUN JSON ===", flush=True)
    print(run_json.read_text(), flush=True)

print("=== ALL TASKS COMPLETED SUCCESSFULLY ===", flush=True)
'''
    runner_path = ROOT / ".cache/run_phase3_remote.py"
    runner_path.write_text(runner_code)
    print(f"Wrote {runner_path} ({runner_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
