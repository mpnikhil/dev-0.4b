"""Evaluate a trained Jev checkpoint or baseline on the CodeSearchNet human benchmark.

Compares model reranking performance against deterministic BM25 baseline.
"""
import argparse
import json
from pathlib import Path
from dev.inference import Predictor
from dev.retrieval import BM25, ndcg

ROOT = Path(__file__).resolve().parents[1]


def evaluate_bm25(benchmark_data):
    results = {}
    for split in ["development", "test"]:
        split_queries = [q for q in benchmark_data if q["split"] == split]
        scores = []
        for q in split_queries:
            codes = [c["code"] for c in q["candidates"]]
            rels = [c["relevance"] for c in q["candidates"]]
            index = BM25(codes)
            pred_scores = index.scores(q["query"])
            order = sorted(range(len(pred_scores)), key=lambda i: -pred_scores[i])
            val = ndcg(rels, order, k=10)
            if val is not None:
                scores.append(val)
        results[split] = {
            "queries": len(split_queries),
            "mean_ndcg_at_10": sum(scores) / len(scores) if scores else 0.0
        }
    return results


def evaluate_model(predictor, benchmark_data, scoring_mode="score"):
    results = {}
    for split in ["development", "test"]:
        split_queries = [q for q in benchmark_data if q["split"] == split]
        ndcg_scores = []
        for q in split_queries:
            query_text = q["query"]
            codes = [c["code"] for c in q["candidates"]]
            rels = [c["relevance"] for c in q["candidates"]]

            candidate_scores = []
            for code in codes:
                try:
                    if scoring_mode == "noul":
                        questions = {
                            "match": {
                                "type": "noul",
                                "instructions": f"Does this code implement the following functionality: {query_text}?"
                            }
                        }
                        ans = predictor.answer(state=code, questions=questions, command="git grep")
                        candidate_scores.append(ans["match"]["probability_yes"])
                    else:  # score
                        questions = {
                            "match": {
                                "type": "score",
                                "instructions": f"How closely does this code match the requested functionality: {query_text}?",
                                "criteria": [
                                    "No relevance / shares keywords only",
                                    "Partially relevant context",
                                    "Direct implementation of requested functionality"
                                ]
                            }
                        }
                        ans = predictor.answer(state=code, questions=questions, command="git grep")
                        candidate_scores.append(ans["match"]["value"])
                except Exception:
                    candidate_scores.append(0.0)

            order = sorted(range(len(candidate_scores)), key=lambda i: -candidate_scores[i])
            val = ndcg(rels, order, k=10)
            if val is not None:
                ndcg_scores.append(val)

        results[split] = {
            "queries": len(split_queries),
            "mean_ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="Path to checkpoint directory (optional)")
    p.add_argument("--benchmark", default="data/benchmarks/csn-human-python.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--mode", default="score", choices=["score", "noul"])
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--output", default="docs/csn-benchmark-evaluation.json")
    args = p.parse_args()

    bench_path = ROOT / args.benchmark
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {bench_path}")

    bench_data = [json.loads(line) for line in bench_path.read_text().splitlines() if line.strip()]

    print("Evaluating BM25 Baseline...")
    bm25_results = evaluate_bm25(bench_data)
    print(f"BM25 Dev NDCG@10:  {bm25_results['development']['mean_ndcg_at_10']:.4f}")
    print(f"BM25 Test NDCG@10: {bm25_results['test']['mean_ndcg_at_10']:.4f}")

    report = {
        "benchmark": "CodeSearchNet Python Human Judgments",
        "total_queries": len(bench_data),
        "bm25_baseline": bm25_results
    }

    if args.checkpoint:
        predictor = Predictor(args.checkpoint, device=args.device)
        if args.max_length:
            predictor.max_length = args.max_length
        model_results = evaluate_model(predictor, bench_data, scoring_mode=args.mode)
        print(f"Model Dev NDCG@10:  {model_results['development']['mean_ndcg_at_10']:.4f}")
        print(f"Model Test NDCG@10: {model_results['test']['mean_ndcg_at_10']:.4f}")
        report["model_evaluation"] = {
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "results": model_results
        }

    out_file = ROOT / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
