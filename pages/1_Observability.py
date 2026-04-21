"""
Observability & Admin Panel — Streamlit page.

Shows recent requests, routing decisions, latency distributions,
error logs, and live aggregate metrics so failures can be diagnosed
and explained in the project report.
"""
import json

import pandas as pd
import streamlit as st

from core.logger import get_recent_requests, init_db
from core.metrics import compute_latency_stats, get_live_metrics

init_db()

st.set_page_config(
    page_title="Observability — Smart Doc Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Observability Panel")
st.markdown(
    "Full audit trail of every request: routing decisions, confidence scores, "
    "latency, tool outputs, and errors."
)

# ── Live metrics ───────────────────────────────────────────────────────────────
metrics = get_live_metrics()
summary = metrics.get("summary", {})
latency = metrics.get("latency", {})

if not summary:
    st.info(
        "No requests logged yet.  \n"
        "Head to the main page, paste some text, and hit **Analyze**."
    )
    st.stop()

# Top KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Requests", summary.get("total_requests", 0))
k2.metric("Avg Latency", f"{summary.get('avg_total_latency_ms', 0):.0f} ms")
k3.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.2f}")
k4.metric("Error Rate", f"{summary.get('error_rate_pct', 0):.1f}%")
k5.metric("Clarification Rate", f"{summary.get('clarification_rate_pct', 0):.1f}%")
k6.metric("👍 Thumbs Up", summary.get("thumbs_up_count", 0))

st.divider()

# ── Charts + latency stats ─────────────────────────────────────────────────────
col_chart, col_lat = st.columns([2, 1])

with col_chart:
    st.subheader("Tool Distribution")
    dist = summary.get("tool_distribution", {})
    if dist:
        df_dist = pd.DataFrame(
            [{"Tool": k, "Requests": v} for k, v in dist.items()]
        ).sort_values("Requests", ascending=True)
        st.bar_chart(df_dist.set_index("Tool"), height=280)

with col_lat:
    st.subheader("Latency Stats (ms)")
    if latency.get("n", 0) > 0:
        st.markdown(
            f"""
| Stat | Value |
|------|-------|
| Average | **{latency['avg_ms']} ms** |
| Median (p50) | {latency['p50_ms']} ms |
| Min | {latency['min_ms']} ms |
| Max | {latency['max_ms']} ms |
| Samples | {latency['n']} |
"""
        )
    else:
        st.info("Not enough successful requests for latency stats.")

st.divider()

# ── Request table ──────────────────────────────────────────────────────────────
st.subheader("Recent Requests")

requests = get_recent_requests(limit=100)

if requests:
    df = pd.DataFrame(requests)

    # Build a clean display frame
    display = pd.DataFrame(
        {
            "Time": df["timestamp"].str[:19].str.replace("T", " "),
            "Tool": df["selected_tool"].fillna("—"),
            "Conf": df["confidence"].round(2).fillna(0),
            "Clarif?": df["needs_clarification"].map({0: "No", 1: "Yes"}),
            "Total ms": df["total_latency_ms"].round(0).fillna(0).astype(int),
            "Feedback": df["user_feedback"].map({1: "👍", -1: "👎", 0: "—"}),
            "Error?": df["error"].apply(lambda x: "❌" if x else ""),
            "Input (preview)": df["input_text"].str[:70] + "…",
        }
    )

    # Highlight errors in red
    def highlight_errors(row):
        return ["background-color: #fee2e2" if row["Error?"] == "❌" else ""] * len(row)

    st.dataframe(
        display.style.apply(highlight_errors, axis=1),
        use_container_width=True,
        height=380,
    )

st.divider()

# ── Detail inspector ───────────────────────────────────────────────────────────
st.subheader("Request Detail Inspector")

if requests:
    labels = [
        f"[{r['timestamp'][:19].replace('T',' ')}]  {r.get('selected_tool','N/A')!s:30s}"
        f"  conf={r.get('confidence') or 0:.2f}  —  {r['input_text'][:55]}…"
        for r in requests
    ]

    idx = st.selectbox("Select a request:", range(len(requests)), format_func=lambda i: labels[i])
    r = requests[idx]

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Input Text**")
        st.text_area("input", value=r["input_text"], height=160, disabled=True, label_visibility="collapsed")
        st.markdown("**Router Rationale**")
        st.info(r.get("rationale") or "—")
        st.markdown("**Clarification Question**")
        st.write(r.get("clarification_question") or "—")  # not stored directly but shown from rationale

    with d2:
        st.markdown("**Tool Output**")
        st.text_area("output", value=r.get("tool_output") or "—", height=160, disabled=True, label_visibility="collapsed")
        st.markdown("**Full Log Record**")
        safe = {k: v for k, v in r.items() if k not in ("input_text", "tool_output")}
        st.code(json.dumps(safe, indent=2), language="json")

    if r.get("error"):
        st.error(f"**Error recorded:** {r['error']}")
