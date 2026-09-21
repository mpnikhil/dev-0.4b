"""Upload chunked model.safetensors to Colab session in parallel."""
import argparse
import concurrent.futures
from pathlib import Path
import time
from colab_cli.state import StateStore
from colab_cli.contents import ContentsClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="jev-a100-v3", help="Colab session name")
    p.add_argument("--input-dir", default=".cache/upload-chunks", help="Local directory containing chunks")
    p.add_argument("--remote-dir", default="/content/chunks", help="Remote directory on Colab")
    p.add_argument("--workers", type=int, default=6, help="Parallel upload workers")
    args = p.parse_args()

    store = StateStore()
    session = store.get(args.session)
    if not session:
        raise ValueError(f"Session '{args.session}' not found in StateStore")
    client = ContentsClient(session)

    chunk_dir = ROOT / args.input_dir
    chunks = sorted([p for p in chunk_dir.glob("part_*") if p.is_file()])
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    print(f"Found {len(chunks)} chunks to upload to {args.session}:{args.remote_dir}")

    # Ensure remote directory exists by touching a dummy file or creating via python
    # Client upload will create parent directories if server supports, or we ensure via colab exec

    def upload_one(chunk_path: Path):
        remote_path = f"{args.remote_dir}/{chunk_path.name}"
        t0 = time.time()
        for attempt in range(3):
            try:
                client.upload(str(chunk_path), remote_path)
                elapsed = time.time() - t0
                size_mb = chunk_path.stat().st_size / 1024 / 1024
                print(f"  Uploaded {chunk_path.name} ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)
                return chunk_path.name
            except Exception as e:
                print(f"  [Attempt {attempt+1}] Error uploading {chunk_path.name}: {e}", flush=True)
                if attempt == 2:
                    return None
                time.sleep(1)

    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(upload_one, chunks))

    success = [r for r in results if r is not None]
    total_elapsed = time.time() - start_all
    print(f"\nUploaded {len(success)}/{len(chunks)} chunks in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if len(success) != len(chunks):
        raise RuntimeError("Some chunks failed to upload")


if __name__ == "__main__":
    main()
