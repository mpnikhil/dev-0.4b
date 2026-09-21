"""Build a Colab A100 training script for Phase 3.

Embeds: source code + v3 datasets + benchmark data + community benchmarks script.
The 1.5 GB Phase 2 checkpoint (model.safetensors) is NOT embedded — it's uploaded
via Google Drive or split-chunk upload.
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    train_file = ROOT / "data/train_v3.jsonl"
    val_file = ROOT / "data/validation_v3.jsonl"
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("data/train_v3.jsonl or data/validation_v3.jsonl not found")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        # Source code
        for py_file in (ROOT / "src/dev").glob("*.py"):
            tar.add(py_file, arcname=f"src/dev/{py_file.name}")
        # Phase 3 datasets
        tar.add(train_file, arcname="data/train_v3.jsonl")
        tar.add(val_file, arcname="data/validation_v3.jsonl")
        # CSN benchmark data + eval script
        csn_bench = ROOT / "data/benchmarks/csn-human-python.jsonl"
        if csn_bench.exists():
            tar.add(csn_bench, arcname="data/benchmarks/csn-human-python.jsonl")
        eval_script = ROOT / "scripts/evaluate_csn_benchmark.py"
        if eval_script.exists():
            tar.add(eval_script, arcname="scripts/evaluate_csn_benchmark.py")
        # Community benchmarks eval script
        comm_script = ROOT / "scripts/evaluate_community_benchmarks.py"
        if comm_script.exists():
            tar.add(comm_script, arcname="scripts/evaluate_community_benchmarks.py")
        # Phase 2 checkpoint metadata (small files only, NOT model.safetensors)
        chk_dir = ROOT / "runs/jev-modernbert-large-a100"
        for item in ["run.json"]:
            p = chk_dir / item
            if p.exists():
                tar.add(p, arcname=f"runs/phase2-checkpoint/{item}")
        # Encoder config
        enc_config = chk_dir / "encoder/config.json"
        if enc_config.exists():
            tar.add(enc_config, arcname="runs/phase2-checkpoint/encoder/config.json")
        # Tokenizer files
        tok_dir = chk_dir / "tokenizer"
        if tok_dir.exists():
            for f in tok_dir.iterdir():
                tar.add(f, arcname=f"runs/phase2-checkpoint/tokenizer/{f.name}")

    payload_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload_mb = len(buffer.getvalue()) / 1024 / 1024
    print(f"Payload size: {payload_mb:.1f} MB")

    remote_code = f'''# Phase 3 — Colab A100 Training Runner
# Warm-start from Phase 2 checkpoint with expanded dataset (25,228 train / 4,285 val)
import base64, io, os, subprocess, sys, tarfile, time, json
from pathlib import Path

print("=== 1. System & GPU Check ===")
import torch
print(f"PyTorch: {{torch.__version__}}, CUDA Available: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"Device: {{torch.cuda.get_device_name()}}, VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}} GB")
    print(f"BF16 Supported: {{torch.cuda.is_bf16_supported()}}")
else:
    print("WARNING: No GPU detected! Training will be very slow.")

print("=== 2. Install Dependencies ===")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.48", "safetensors", "numpy", "peft", "datasets"], check=True)

print("=== 3. Extract Payload ===")
workspace = Path("/content/jev-workspace")
workspace.mkdir(parents=True, exist_ok=True)
payload_bytes = base64.b64decode({repr(payload_b64)})
with tarfile.open(fileobj=io.BytesIO(payload_bytes), mode="r:gz") as tar:
    tar.extractall(workspace)

print(f"Extracted files to {{workspace}}:")
for p in sorted(workspace.rglob("*")):
    if p.is_file():
        print(f"  {{p.relative_to(workspace)}} ({{p.stat().st_size:,}} bytes)")

print("=== 4. Upload Phase 2 Checkpoint (model.safetensors) ===")
chk_dir = workspace / "runs/phase2-checkpoint"
chk_dir.mkdir(parents=True, exist_ok=True)
model_weights = chk_dir / "model.safetensors"

if not model_weights.exists():
    # Try Google Drive first
    drive_path = Path("/content/drive/MyDrive/jev-checkpoints/model.safetensors")
    if drive_path.exists():
        import shutil
        shutil.copy2(drive_path, model_weights)
        print(f"Copied checkpoint from Google Drive ({{model_weights.stat().st_size:,}} bytes)")
    else:
        # Try reassembling from chunks
        chunks_dir = Path("/content/chunks")
        if chunks_dir.exists() and list(chunks_dir.glob("part_*")):
            print("Reassembling from uploaded chunks...")
            parts = sorted(chunks_dir.glob("part_*"))
            with open(model_weights, "wb") as out:
                for part in parts:
                    out.write(part.read_bytes())
            print(f"Reassembled checkpoint ({{model_weights.stat().st_size:,}} bytes)")
        else:
            # Fall back to Colab file upload
            print("Please upload model.safetensors (1.5 GB) or upload chunks to /content/chunks/")
            from google.colab import files
            uploaded = files.upload()
            for name, data in uploaded.items():
                if "safetensors" in name:
                    model_weights.write_bytes(data)
                    print(f"Saved uploaded checkpoint ({{len(data):,}} bytes)")

if model_weights.exists():
    print(f"Checkpoint ready: {{model_weights}} ({{model_weights.stat().st_size:,}} bytes)")
else:
    raise FileNotFoundError("model.safetensors not found! Cannot proceed without Phase 2 checkpoint.")

print("=== 5. Launch Phase 3 Training ===")
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
print(f"Command: {{' '.join(cmd)}}")
start_t = time.time()
res = subprocess.run(cmd, env=env, cwd=str(workspace))
elapsed = time.time() - start_t
print(f"Training exit code: {{res.returncode}}, elapsed: {{elapsed:.1f}}s ({{elapsed/60:.1f}} min)")
if res.returncode != 0:
    print("ERROR: Training failed!")
    sys.exit(res.returncode)

print("=== 6. Run CSN Benchmark Evaluation ===")
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
    eval_res = subprocess.run(bench_cmd, env=env, cwd=str(workspace))
    print(f"CSN benchmark exit code: {{eval_res.returncode}}")
else:
    print("Skipping CSN benchmark (data not found)")

print("=== 7. Run Community Benchmarks ===")
comm_script = workspace / "scripts/evaluate_community_benchmarks.py"
if comm_script.exists():
    comm_cmd = [
        sys.executable, str(comm_script),
        "--checkpoint", str(workspace / "runs/jev-phase3-a100"),
        "--device", "cuda",
        "--samples", "500",
        "--output", str(workspace / "runs/jev-phase3-a100/community-benchmark-evaluation.json")
    ]
    comm_res = subprocess.run(comm_cmd, env=env, cwd=str(workspace))
    print(f"Community benchmark exit code: {{comm_res.returncode}}")
else:
    print("Skipping community benchmarks (script not found)")

print("=== 8. Prepare Download Chunks ===")
out_dir = workspace / "runs/jev-phase3-a100"
chunks_dir = Path("/content/phase3-chunks")
chunks_dir.mkdir(parents=True, exist_ok=True)
subprocess.run(["split", "-b", "20M", "-d",
                 str(out_dir / "model.safetensors"),
                 str(chunks_dir / "part_")], check=True)
n_chunks = len(list(chunks_dir.glob("part_*")))
print(f"Created {{n_chunks}} chunks in {{chunks_dir}}")

# Also create inference archive (without trainer.pt)
inf_archive = Path("/content/jev-phase3-inference.tar.gz")
with tarfile.open(inf_archive, "w:gz") as tar:
    for item in out_dir.glob("*"):
        if item.name != "trainer.pt":
            tar.add(item, arcname=f"jev-phase3-a100/{{item.name}}")
print(f"Inference archive: {{inf_archive}} ({{inf_archive.stat().st_size:,}} bytes)")

# Print final run summary
run_json = out_dir / "run.json"
if run_json.exists():
    run_data = json.loads(run_json.read_text())
    print("\\n=== FINAL RESULTS ===")
    print(json.dumps(run_data, indent=2))

print("\\n=== DONE! Download chunks from /content/phase3-chunks/ ===")
'''

    out_script = ROOT / ".cache/train_colab_phase3.py"
    out_script.parent.mkdir(exist_ok=True)
    out_script.write_text(remote_code)
    print(f"Wrote: {out_script} ({out_script.stat().st_size:,} bytes)")
    print(f"\nNext steps:")
    print(f"  1. Open a new Colab A100 session")
    print(f"  2. Upload model.safetensors chunks to /content/chunks/")
    print(f"  3. Paste & run the script from {out_script}")


if __name__ == "__main__":
    main()
