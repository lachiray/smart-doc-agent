"""
Evaluation script — runs the agent router over labeled test cases and
reports routing accuracy and average latency.

Usage:
    cd smart-doc-assistant
    python evaluation/run_evaluation.py

Output:
    evaluation/results/eval_results.json   — full per-case results
    Prints a summary table to stdout
"""
import json
import os
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from dotenv import load_dotenv

from agent.router import route_input
from core.metrics import compute_routing_accuracy

load_dotenv()

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
RESULTS_PATH = Path(__file__).parent / "results" / "eval_results.json"


def run() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    with open(TEST_CASES_PATH) as f:
        test_cases: list[dict] = json.load(f)

    results: list[dict] = []
    print(f"\nRunning evaluation on {len(test_cases)} test cases…\n{'─' * 62}")

    for i, case in enumerate(test_cases, start=1):
        case_id = case["id"]
        expected_tool = case["expected_tool"]
        expects_clarif = case.get("expects_clarification", False)

        print(f"[{i:02d}/{len(test_cases)}] {case_id}")

        try:
            router_out, latency = route_input(case["input_text"], client)

            tool_correct = router_out.selected_tool == expected_tool
            # Clarification is correct if both agree, or if expected=True and conf<0.6
            clari_correct = router_out.needs_clarification == expects_clarif

            verdict = "✓" if tool_correct else "✗"
            print(
                f"       {verdict}  expected={expected_tool:25s}  "
                f"got={router_out.selected_tool:25s}  "
                f"conf={router_out.confidence:.2f}  {latency*1000:.0f}ms"
            )

            results.append(
                {
                    "id": case_id,
                    "description": case.get("description", ""),
                    "expected_tool": expected_tool,
                    "actual_tool": router_out.selected_tool,
                    "tool_correct": tool_correct,
                    "expected_clarification": expects_clarif,
                    "actual_clarification": router_out.needs_clarification,
                    "clarification_correct": clari_correct,
                    "confidence": router_out.confidence,
                    "rationale": router_out.rationale,
                    "latency_ms": round(latency * 1000, 1),
                    "notes": case.get("notes", ""),
                    "error": None,
                }
            )

        except Exception as exc:
            print(f"       ✗  ERROR: {exc}")
            results.append(
                {
                    "id": case_id,
                    "expected_tool": expected_tool,
                    "actual_tool": None,
                    "tool_correct": False,
                    "error": str(exc),
                }
            )

        # Gentle rate limiting
        time.sleep(0.3)

    # ── Metrics ────────────────────────────────────────────────────────────────
    accuracy_data = compute_routing_accuracy(
        [{"expected_tool": r["expected_tool"], "actual_tool": r.get("actual_tool")} for r in results]
    )

    success_results = [r for r in results if r.get("latency_ms") is not None]
    avg_latency = (
        sum(r["latency_ms"] for r in success_results) / len(success_results)
        if success_results
        else 0
    )

    misrouted = [r for r in results if not r.get("tool_correct")]
    correct_clari = [r for r in results if r.get("clarification_correct", False)]

    summary = {
        "routing_accuracy": accuracy_data["accuracy"],
        "correct": accuracy_data["correct"],
        "total": accuracy_data["total"],
        "avg_latency_ms": round(avg_latency, 1),
        "clarification_accuracy": round(len(correct_clari) / len(results), 3) if results else 0,
        "misrouted_cases": [r["id"] for r in misrouted],
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print report ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"  ROUTING ACCURACY      : {accuracy_data['accuracy']:.1%}  "
          f"({accuracy_data['correct']}/{accuracy_data['total']} correct)")
    print(f"  AVG ROUTER LATENCY    : {avg_latency:.0f} ms")
    print(f"  CLARIFICATION ACCURACY: {summary['clarification_accuracy']:.1%}")
    print(f"{'═' * 62}")

    if misrouted:
        print(f"\nMisrouted cases ({len(misrouted)}):")
        for r in misrouted:
            print(
                f"  • {r['id']}\n"
                f"      expected : {r['expected_tool']}\n"
                f"      got      : {r.get('actual_tool', 'ERROR')}\n"
                f"      conf     : {r.get('confidence', '?')}\n"
                f"      rationale: {r.get('rationale', r.get('error', '?'))}"
            )

    print(f"\nFull results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    run()
