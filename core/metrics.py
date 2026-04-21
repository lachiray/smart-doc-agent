"""
Metrics module.

Tracks two primary metrics:
  1. Quality metric   — routing accuracy on a labeled evaluation set
  2. Operational metric — end-to-end latency statistics (avg, min, max, p50)

Additional metrics available from the logger summary:
  clarification rate, error rate, thumbs-up rate
"""
from typing import Optional

from core.logger import get_recent_requests, get_metrics_summary


# ── Quality metric ─────────────────────────────────────────────────────────────

def compute_routing_accuracy(labeled_results: list[dict]) -> dict:
    """
    Quality Metric: fraction of test cases where the router chose the expected tool.

    labeled_results: list of dicts with keys 'expected_tool' and 'actual_tool'.
    """
    if not labeled_results:
        return {"accuracy": None, "correct": 0, "total": 0}

    correct = sum(
        1 for r in labeled_results
        if r.get("expected_tool") == r.get("actual_tool")
    )
    return {
        "accuracy": round(correct / len(labeled_results), 3),
        "correct": correct,
        "total": len(labeled_results),
    }


# ── Operational metric ─────────────────────────────────────────────────────────

def compute_latency_stats(requests: Optional[list] = None) -> dict:
    """
    Operational Metric: end-to-end latency across successful requests.

    If requests is None, fetches the last 100 from the database.
    """
    if requests is None:
        requests = get_recent_requests(limit=100)

    latencies = [
        r["total_latency_ms"]
        for r in requests
        if r.get("total_latency_ms") is not None and r.get("error") is None
    ]

    if not latencies:
        return {"avg_ms": None, "min_ms": None, "max_ms": None, "p50_ms": None, "n": 0}

    sorted_lat = sorted(latencies)
    return {
        "avg_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(sorted_lat[0], 1),
        "max_ms": round(sorted_lat[-1], 1),
        "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 1),
        "n": len(latencies),
    }


def get_live_metrics() -> dict:
    """Combine all metrics for the observability dashboard."""
    return {
        "summary": get_metrics_summary(),
        "latency": compute_latency_stats(),
    }
