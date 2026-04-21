"""
Smart Document Action Assistant — main Streamlit application.

AGENTIC DESIGN OVERVIEW
───────────────────────
This app demonstrates agentic behavior by using an LLM as a router that
dynamically decides WHICH tool to invoke based on the user's input.

Flow:
  1. User submits text (paste or file upload)
  2. agent/router.py sends the text to Claude with a routing prompt
     → Claude returns structured JSON: {selected_tool, confidence, rationale, ...}
  3. app.py reads the router decision and dispatches to the chosen tool
     (never runs more than one tool per request)
  4. agent/tools.py executes the selected tool with a targeted prompt
  5. Results + a visible trace are rendered to the user
  6. The request is logged to SQLite for observability

The router is the only decision-maker. The app itself is policy-free.
"""

import json
import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from agent.router import route_input
from agent.tools import TOOL_ICONS, TOOL_LABELS, TOOL_REGISTRY
from core.client import get_client as _get_client
from core.logger import init_db, log_request, update_feedback

load_dotenv()
init_db()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Doxy ✨ your doc assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hero header */
.hero-wrap {
    background: linear-gradient(135deg, #f0e7ff 0%, #fce4f5 50%, #ffe8d6 100%);
    border-radius: 20px;
    padding: 2rem 2.4rem 1.6rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 24px rgba(167,139,250,0.12);
}
.main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #7c3aed, #db2777, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 0.4rem;
}
.sub-title { font-size: 1rem; color: #6b7280; margin: 0; }

/* Section headings */
.section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #a78bfa; margin-bottom: 0.5rem;
}

/* Tool badge pill */
.tool-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 999px;
    background: linear-gradient(135deg, #ede9fe, #fce7f3);
    color: #7c3aed; font-weight: 700; font-size: 0.92rem;
    border: 1.5px solid #ddd6fe;
    box-shadow: 0 2px 8px rgba(124,58,237,0.10);
}

/* Confidence bar */
.conf-wrap { margin: 6px 0; }
.conf-track {
    height: 8px; background: #f3e8ff; border-radius: 999px; overflow: hidden;
}
.conf-fill  { height: 8px; border-radius: 999px; }

/* Result card */
.result-card {
    background: linear-gradient(135deg, #faf5ff 0%, #fdf2f8 100%);
    border: 1.5px solid #e9d5ff;
    padding: 1.2rem 1.5rem; border-radius: 16px;
    margin-top: 0.5rem; line-height: 1.8;
    box-shadow: 0 2px 12px rgba(167,139,250,0.08);
}

/* How-it-works card */
.how-card {
    background: linear-gradient(135deg, #fefce8 0%, #f0fdf4 100%);
    border: 1.5px solid #d9f99d; border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 10px rgba(134,239,172,0.12);
}

/* Tool table rows */
.tool-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #f3e8ff;
}
.tool-row:last-child { border-bottom: none; }
.tool-icon { font-size: 1.15rem; width: 28px; text-align: center; flex-shrink: 0; }
.tool-name { font-weight: 700; font-size: 0.88rem; color: #4c1d95; }
.tool-desc { font-size: 0.82rem; color: #6b7280; }

/* Clarification banner */
.clari-box {
    background: linear-gradient(135deg, #fefce8, #fff7ed);
    border: 1.5px solid #fcd34d; border-radius: 14px;
    padding: 1rem 1.2rem; margin-top: 1rem;
}

/* Confidence label colours */
.conf-high   { color: #059669; font-weight: 700; }
.conf-medium { color: #d97706; font-weight: 700; }
.conf-low    { color: #dc2626; font-weight: 700; }

/* Divider */
hr { border: none; border-top: 2px dashed #e9d5ff; margin: 1.4rem 0; }

/* Streamlit button overrides */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.28) !important;
    transition: opacity 0.15s !important;
}
div.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_client():
    try:
        return _get_client()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()


def render_confidence(confidence: float) -> None:
    pct = int(confidence * 100)
    if pct >= 80:
        color = "#22c55e"
        label = "High"
    elif pct >= 55:
        color = "#f59e0b"
        label = "Medium"
    else:
        color = "#ef4444"
        label = "Low"
    st.markdown(
        f"""<div class="conf-wrap">
        <div style="font-size:0.75rem; color:#64748b; margin-bottom:3px">
            Confidence: <strong style="color:{color}">{pct}% ({label})</strong>
        </div>
        <div class="conf-track">
          <div class="conf-fill" style="width:{pct}%; background:{color}"></div>
        </div></div>""",
        unsafe_allow_html=True,
    )


SAMPLE_TEXTS = {
    "Meeting notes with tasks": (
        "Q3 Planning Meeting — 2024-09-10\n\n"
        "Attendees: Sarah (PM), Dev team, Marketing\n\n"
        "Decisions:\n"
        "- Sarah will finalize the roadmap document by Friday Sep 13.\n"
        "- Dev team to complete API integration by end of sprint (Sep 20).\n"
        "- Marketing to prepare launch copy draft — due Sep 18, send to Sarah for review.\n"
        "- John to schedule user testing sessions next week and share calendar invite.\n"
        "- Budget approval needed from CFO before Oct 1 to proceed with vendor contracts."
    ),
    "Policy excerpt with risks": (
        "Section 4.2 — Data Retention and Deletion Policy\n\n"
        "All customer personal data must be retained for a minimum of 5 years in accordance "
        "with applicable financial regulations. However, upon customer request, personal "
        "data shall be deleted within 30 days, notwithstanding any conflicting retention "
        "requirements. The company shall not be liable for data breaches arising from "
        "third-party processor failures provided reasonable due diligence was performed. "
        "Employees who violate this policy may face disciplinary action up to and including "
        "termination. The company reserves the right to modify this policy at any time "
        "without prior notice to customers."
    ),
    "Informational briefing": (
        "Background Briefing: Large Language Models in Enterprise\n\n"
        "Large language models (LLMs) have seen rapid adoption across enterprise verticals "
        "in 2023–2024. These systems, trained on vast text corpora, are now deployed for "
        "customer support automation, code generation, document analysis, and knowledge "
        "management. Key vendors include OpenAI (GPT-4), Anthropic (Claude), and Google "
        "(Gemini). Enterprises must weigh capability, data privacy, cost, and vendor lock-in "
        "when selecting a provider. Fine-tuning and retrieval-augmented generation (RAG) "
        "are the dominant approaches for adapting base models to specific business contexts."
    ),
    "Ambiguous short input": "Can you check this for me?",
}


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-wrap">
  <p class="main-title">✨ meet doxy</p>
  <p class="sub-title">drop in any document — email, meeting notes, policy, report — and doxy figures out exactly what to do with it 🪄</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")

# ── Input column ───────────────────────────────────────────────────────────────
with left:
    st.markdown('<p class="section-label">📄 your document</p>', unsafe_allow_html=True)

    # Sample loader
    sample_choice = st.selectbox(
        "Load a sample input (optional):",
        ["— select a sample —"] + list(SAMPLE_TEXTS.keys()),
    )
    prefill = SAMPLE_TEXTS.get(sample_choice, "")

    # File upload
    uploaded = st.file_uploader(
        "Or upload a .txt or .pdf file:",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )
    file_text = ""
    if uploaded:
        if uploaded.type == "text/plain":
            file_text = uploaded.read().decode("utf-8", errors="replace")
        elif uploaded.type == "application/pdf":
            try:
                import pypdf  # type: ignore

                reader = pypdf.PdfReader(uploaded)
                file_text = "\n\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                st.success(f"Extracted {len(reader.pages)} page(s) from PDF.")
            except ImportError:
                st.warning("PDF support requires `pypdf`. Run: `pip install pypdf`")

    # Text area — prefer file content, then sample, then empty
    default_text = file_text or prefill
    user_text = st.text_area(
        "Paste text here:",
        value=default_text,
        height=230,
        placeholder="Paste an email, meeting notes, policy excerpt, contract, or any document…",
        label_visibility="collapsed",
    )

    # Clarification answer field (shown when router asks a follow-up)
    clarification_answer = ""
    if st.session_state.get("pending_clarification"):
        st.markdown(
            f'<div class="clari-box">🤔 <strong>doxy has a quick question:</strong><br>'
            f'{st.session_state.pending_clarification}</div>',
            unsafe_allow_html=True,
        )
        clarification_answer = st.text_input(
            "Your answer:", key="clarification_input", placeholder="Type your answer…"
        )

    run_btn = st.button("✨ analyze with doxy", type="primary", use_container_width=True)

# ── Info column ────────────────────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-label">🧠 how doxy thinks</p>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="how-card">
<p style="font-size:0.88rem; color:#374151; margin-bottom:0.9rem;">
doxy uses a smart <strong>agent router</strong> — not a rigid pipeline — to pick the best action for your doc every time 🎯
</p>

<div class="tool-row">
  <span class="tool-icon">📋</span>
  <div><div class="tool-name">Summarize</div><div class="tool-desc">long or informational content</div></div>
</div>
<div class="tool-row">
  <span class="tool-icon">✅</span>
  <div><div class="tool-name">Extract Action Items</div><div class="tool-desc">tasks, deadlines &amp; owners</div></div>
</div>
<div class="tool-row">
  <span class="tool-icon">🏷️</span>
  <div><div class="tool-name">Classify</div><div class="tool-desc">ambiguous or mixed-genre docs</div></div>
</div>
<div class="tool-row">
  <span class="tool-icon">🚨</span>
  <div><div class="tool-name">Risk Scan</div><div class="tool-desc">legal, compliance &amp; safety flags</div></div>
</div>
<div class="tool-row">
  <span class="tool-icon">🔍</span>
  <div><div class="tool-name">Retrieve Context</div><div class="tool-desc">questions needing background</div></div>
</div>

<p style="font-size:0.8rem; color:#9ca3af; margin-top:0.9rem; margin-bottom:0;">
💬 if doxy isn't sure (confidence &lt; 0.6) she'll ask before guessing
</p>
</div>

<p style="font-size:0.8rem; color:#a78bfa; margin-top:0.9rem;">📊 check the <strong>Observability</strong> page for logs &amp; metrics</p>
""",
        unsafe_allow_html=True,
    )

st.divider()

# ── Processing ─────────────────────────────────────────────────────────────────
if run_btn:
    # Merge clarification into text if provided
    working_text = user_text.strip()
    if clarification_answer.strip():
        working_text += f"\n\n[User clarification: {clarification_answer.strip()}]"
        st.session_state.pending_clarification = None

    if len(working_text) < 15:
        st.warning("Please enter at least a few words to analyze.")
        st.stop()

    client = get_client()

    # ── Step 1: Route ──────────────────────────────────────────────────────────
    with st.spinner("🧠 doxy is reading your doc…"):
        try:
            router_result, router_latency = route_input(working_text, client)
        except Exception as exc:
            st.error(f"Router failed: {exc}")
            log_request(working_text, None, None, False, None, None, 0, 0, error=str(exc))
            st.stop()

    # ── Clarification path ─────────────────────────────────────────────────────
    if router_result.needs_clarification and not clarification_answer.strip():
        st.session_state.pending_clarification = router_result.clarification_question
        log_request(
            working_text, None, router_result.confidence,
            True, router_result.rationale, None,
            router_latency * 1000, 0,
        )
        st.info(
            f"**doxy needs a little more info before diving in ✨**\n\n"
            f"*{router_result.rationale}*"
        )
        st.rerun()

    # ── Step 2: Execute selected tool ──────────────────────────────────────────
    tool_fn = TOOL_REGISTRY[router_result.selected_tool]
    tool_output = ""
    tool_latency = 0.0
    error_msg = None

    tool_label = TOOL_LABELS[router_result.selected_tool]
    tool_icon = TOOL_ICONS[router_result.selected_tool]

    with st.spinner(f"{tool_icon} Running **{tool_label}**…"):
        try:
            tool_output, tool_latency = tool_fn(working_text, client)
        except Exception as exc:
            error_msg = str(exc)
            tool_output = f"*Tool error: {exc}*"

    # ── Log ────────────────────────────────────────────────────────────────────
    log_row_id = log_request(
        input_text=working_text,
        selected_tool=router_result.selected_tool,
        confidence=router_result.confidence,
        needs_clarification=router_result.needs_clarification,
        rationale=router_result.rationale,
        tool_output=tool_output,
        router_latency_ms=router_latency * 1000,
        tool_latency_ms=tool_latency * 1000,
        error=error_msg,
    )
    st.session_state.last_log_id = log_row_id

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">🎉 doxy\'s verdict</p>', unsafe_allow_html=True)

    header_left, header_mid, header_right = st.columns([3, 3, 1])

    with header_left:
        st.markdown(
            f'<span class="tool-badge">{tool_icon} {tool_label}</span>',
            unsafe_allow_html=True,
        )

    with header_mid:
        render_confidence(router_result.confidence)

    with header_right:
        total_ms = (router_latency + tool_latency) * 1000
        st.metric("Time", f"{total_ms:.0f} ms")

    st.markdown("**💡 why doxy chose this:**")
    st.info(router_result.rationale)

    st.markdown("**📝 output:**")
    st.markdown(
        f'<div class="result-card">{tool_output}</div>',
        unsafe_allow_html=True,
    )

    # ── Feedback ───────────────────────────────────────────────────────────────
    st.markdown(" ")
    fb1, fb2, _ = st.columns([1, 1, 6])
    with fb1:
        if st.button("👍 Helpful", key="thumbsup"):
            update_feedback(log_row_id, 1)
            st.toast("Feedback saved — thank you!")
    with fb2:
        if st.button("👎 Not helpful", key="thumbsdown"):
            update_feedback(log_row_id, -1)
            st.toast("Feedback saved — thank you!")

    # ── Trace panel ────────────────────────────────────────────────────────────
    with st.expander("🔍 Tool Call Trace", expanded=False):
        trace = {
            "steps": [
                {
                    "step": 1,
                    "name": "route_input",
                    "model": "claude-sonnet-4-6",
                    "output": {
                        "selected_tool": router_result.selected_tool,
                        "confidence": router_result.confidence,
                        "needs_clarification": router_result.needs_clarification,
                        "rationale": router_result.rationale,
                    },
                    "latency_ms": round(router_latency * 1000, 1),
                },
                {
                    "step": 2,
                    "name": router_result.selected_tool,
                    "model": "claude-sonnet-4-6",
                    "status": "error" if error_msg else "success",
                    "error": error_msg,
                    "latency_ms": round(tool_latency * 1000, 1),
                },
            ],
            "total_latency_ms": round(total_ms, 1),
            "log_id": log_row_id,
        }
        st.code(json.dumps(trace, indent=2), language="json")

elif not run_btn and st.session_state.get("pending_clarification"):
    # Keep the clarification UI visible if waiting for user answer
    pass
