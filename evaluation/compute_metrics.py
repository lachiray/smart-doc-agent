"""
Compute and display metrics from a saved evaluation results file.

Usage:
    python evaluation/compute_metrics.py
    python evaluation/compute_metrics.py --results evaluation/results/eval_results.json
"""
import argparse
import json
from pathlib import Path


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            "Run evaluation/run_evaluation.py first."
        )
    with open(path) as f:
        return json.load(f)


def print_metrics(data: dict) -> None:
    results = data.get("results", [])
    if not results:
        print("No results to analyze.")
        return

    print("\n" + "═" * 60)
    print("  EVALUATION METRICS REPORT")
    print("═" * 60)

    # ── Quality metric: routing accuracy ──────────────────────────────────────
    acc = data.get("routing_accuracy", 0)
    correct = data.get("correct", 0)
    total = data.get("total", 0)
    print(f"\n  [Quality]  Routing Accuracy : {acc:.1%}  ({correct}/{total})")

    # Per-tool accuracy
    from collections import defaultdict
    tool_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        exp = r.get("expected_tool", "unknown")
        tool_stats[exp]["total"] += 1
        if r.get("tool_correct"):
            tool_stats[exp]["correct"] += 1

    print("\n  Per-tool breakdown:")
    for tool, s in sorted(tool_stats.items()):
        pct = s["correct"] / s["total"] if s["total"] else 0
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"    {tool:28s} {bar} {pct:.0%} ({s['correct']}/{s['total']})")

    # ── Operational metric: latency ────────────────────────────────────────────
    avg_lat = data.get("avg_latency_ms", 0)
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    if latencies:
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p90 = sorted_lat[int(len(sorted_lat) * 0.9)]
        print(f"\n  [Operational] Avg Router Latency : {avg_lat:.0f} ms")
        print(f"                P50               : {p50:.0f} ms")
        print(f"                P90               : {p90:.0f} ms")
        print(f"                Min / Max         : {min(latencies):.0f} / {max(latencies):.0f} ms")

    # ── Clarification accuracy ─────────────────────────────────────────────────
    clari_acc = data.get("clarification_accuracy", 0)
    print(f"\n  [Quality]  Clarification Accuracy : {clari_acc:.1%}")

    # Confidence stats
    confidences = [r["confidence"] for r in results if r.get("confidence") is not None]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"\n  Avg Router Confidence : {avg_conf:.2f}")

    # ── Where it works / fails ─────────────────────────────────────────────────
    correct_cases = [r for r in results if r.get("tool_correct")]
    wrong_cases = [r for r in results if not r.get("tool_correct")]

    print(f"\n  {'─' * 58}")
    print(f"  ✓ WHERE IT WORKS WELL ({len(correct_cases)} cases):")
    for r in correct_cases[:5]:
        print(f"    [{r['id']}] → {r['expected_tool']}  (conf={r.get('confidence', 0):.2f})")

    if wrong_cases:
        print(f"\n  ✗ MISROUTED / FAILURES ({len(wrong_cases)} cases):")
        for r in wrong_cases:
            print(
                f"    [{r['id']}]\n"
                f"      Expected : {r.get('expected_tool')}\n"
                f"      Got      : {r.get('actual_tool', 'ERROR')}\n"
                f"      Conf     : {r.get('confidence', '?')}\n"
                f"      Why      : {r.get('rationale', r.get('error', ''))[:120]}"
            )

    print("\n  RECOMMENDATIONS:")
    if acc < 0.80:
        print("  • Accuracy below 80% — consider adding more routing signal examples to the system prompt.")
    if avg_lat > 3000:
        print("  • Latency above 3s — consider using claude-haiku-4-5 for the routing step.")
    if acc >= 0.85:
        print("  • Good routing accuracy. Focus on improving tool output quality.")

    print("\n" + "═" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="evaluation/results/eval_results.json",
        help="Path to eval_results.json",
    )
    args = parser.parse_args()

    path = Path(args.results)
    try:
        data = load_results(path)
        print_metrics(data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
