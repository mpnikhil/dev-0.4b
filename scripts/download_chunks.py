"""Download chunked model.safetensors from Colab and reconstruct."""
import concurrent.futures
from pathlib import Path
import time
from colab_cli.state import StateStore
from colab_cli.contents import ContentsClient

ROOT = Path(__file__).resolve().parents[1]

import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="jev-a100-v3", help="Session name")
    p.add_argument("--remote-dir", default="/content/phase3-chunks", help="Remote chunk directory")
    p.add_argument("--output", default="runs/jev-phase3-a100/model.safetensors", help="Target output file")
    args = p.parse_args()

    store = StateStore()
    session = store.get(args.session)
    if not session:
        raise ValueError(f"{args.session} session not found in StateStore")
    client = ContentsClient(session)

    chunk_dir = ROOT / "runs/chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically list chunks from remote session
    remote_data = client.list_dir(args.remote_dir)
    chunks = sorted([item["name"] for item in remote_data.get("content", []) if item["name"].startswith("part_")])
    if not chunks:
        # Fallback to checking up to 120 chunks
        chunks = [f"part_{i:02d}" for i in range(120)]

    print(f"Downloading chunks from {args.remote_dir} for {args.session} using ThreadPoolExecutor...")

    def download_one(name):
        remote_path = f"{args.remote_dir}/{name}"
        local_path = chunk_dir / name
        start = time.time()
        try:
            client.download(remote_path, str(local_path))
            print(f"  Downloaded {name} ({local_path.stat().st_size / 1024 / 1024:.1f} MB) in {time.time()-start:.1f}s")
            return name
        except Exception as e:
            return None

    start_all = time.time()
    downloaded_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(download_one, chunks))
        downloaded_chunks = sorted([r for r in results if r is not None])

    print(f"All {len(downloaded_chunks)} chunks downloaded in {time.time()-start_all:.1f}s")

    # Reconstruct model.safetensors
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reconstructing {target}...")
    with open(target, "wb") as outfile:
        for name in downloaded_chunks:
            chunk_file = chunk_dir / name
            outfile.write(chunk_file.read_bytes())

    print(f"Reconstructed {target}: {target.stat().st_size} bytes ({target.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
