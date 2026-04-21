"""
Tool implementations.

Each tool is a targeted LLM prompt optimized for one specific job.
The router in router.py decides which tool to call; this module executes it.
Tools are intentionally simple — the intelligence lives in the router.
"""
import time
from typing import Callable, Union

import anthropic

from core.client import get_model_id

# ── Tool functions ─────────────────────────────────────────────────────────────

def summarize_text(text: str, client: anthropic.Anthropic) -> tuple[str, float]:
    """Produce a structured, concise summary of the input."""
    start = time.perf_counter()
    msg = client.messages.create(
        model=get_model_id(),
        max_tokens=700,
        messages=[
            {
                "role": "user",
                "content": (
                    "Provide a clear, structured summary of the following text.\n\n"
                    "Format your response as:\n"
                    "**Key Points** (3–5 bullets)\n"
                    "**Summary** (2–3 sentences)\n"
                    "**Document Type** (one phrase)\n\n"
                    f"Text:\n{text}"
                ),
            }
        ],
    )
    return msg.content[0].text, time.perf_counter() - start


def extract_action_items(text: str, client: anthropic.Anthropic) -> tuple[str, float]:
    """Extract actionable tasks, owners, and deadlines from the input."""
    start = time.perf_counter()
    msg = client.messages.create(
        model=get_model_id(),
        max_tokens=700,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract all action items, tasks, and deadlines from the text below.\n\n"
                    "For each item provide:\n"
                    "- **Task**: what needs to be done\n"
                    "- **Owner**: who is responsible (if mentioned, else 'Unassigned')\n"
                    "- **Due**: deadline or timeline (if mentioned, else 'Not specified')\n"
                    "- **Priority**: High / Medium / Low based on context signals\n\n"
                    "If no clear action items exist, say so explicitly.\n\n"
                    f"Text:\n{text}"
                ),
            }
        ],
    )
    return msg.content[0].text, time.perf_counter() - start


def classify_text(text: str, client: anthropic.Anthropic) -> tuple[str, float]:
    """Classify the document across multiple dimensions."""
    start = time.perf_counter()
    msg = client.messages.create(
        model=get_model_id(),
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this text across the following dimensions:\n\n"
                    "1. **Document Type** — e.g. email, meeting notes, contract, policy, report, memo\n"
                    "2. **Domain / Category** — e.g. HR, Legal, Finance, Technical, Operations\n"
                    "3. **Tone** — e.g. formal, informal, urgent, neutral, persuasive\n"
                    "4. **Primary Purpose** — e.g. inform, request action, instruct, report, negotiate\n"
                    "5. **Intended Audience** — e.g. executive team, engineering team, general public\n"
                    "6. **Sensitivity Level** — Public / Internal / Confidential / Restricted\n\n"
                    "Be specific and concise for each dimension.\n\n"
                    f"Text:\n{text}"
                ),
            }
        ],
    )
    return msg.content[0].text, time.perf_counter() - start


def risk_scan(text: str, client: anthropic.Anthropic) -> tuple[str, float]:
    """Identify risks, compliance issues, and red flags in the input."""
    start = time.perf_counter()
    msg = client.messages.create(
        model=get_model_id(),
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": (
                    "Scan this text for potential risks, concerns, and issues.\n\n"
                    "Check for:\n"
                    "- Legal or regulatory compliance risks\n"
                    "- Financial or contractual exposure\n"
                    "- Operational or process risks\n"
                    "- Security or privacy concerns\n"
                    "- Reputational risks\n"
                    "- Ambiguous obligations or missing information\n\n"
                    "For each risk found, provide:\n"
                    "- **Risk**: brief description\n"
                    "- **Type**: Legal / Financial / Operational / Security / Reputational\n"
                    "- **Severity**: 🔴 High / 🟡 Medium / 🟢 Low\n"
                    "- **Recommendation**: what to do about it\n\n"
                    "If no significant risks are found, clearly state that.\n\n"
                    f"Text:\n{text}"
                ),
            }
        ],
    )
    return msg.content[0].text, time.perf_counter() - start


def retrieve_context(text: str, client: anthropic.Anthropic) -> tuple[str, float]:
    """
    Retrieve relevant background context for the input.

    In production this would query a vector database or retrieval API.
    Here we leverage the LLM's parametric knowledge to provide relevant context,
    and clearly frame what external sources should be consulted.
    """
    start = time.perf_counter()
    msg = client.messages.create(
        model=get_model_id(),
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": (
                    "The text below references concepts, regulations, or topics that may "
                    "benefit from additional context.\n\n"
                    "Please:\n"
                    "1. **Identify** the key concepts or questions in the text\n"
                    "2. **Provide** relevant background context for each\n"
                    "3. **Note** important related considerations the reader should know\n"
                    "4. **Suggest** what additional sources or documents would be helpful\n\n"
                    "Be concrete and actionable.\n\n"
                    f"Text:\n{text}"
                ),
            }
        ],
    )
    return msg.content[0].text, time.perf_counter() - start


# ── Registry ───────────────────────────────────────────────────────────────────

# Maps tool name → callable; used by app.py to dispatch after routing
TOOL_REGISTRY: dict[str, Callable] = {
    "summarize_text": summarize_text,
    "extract_action_items": extract_action_items,
    "classify_text": classify_text,
    "risk_scan": risk_scan,
    "retrieve_context": retrieve_context,
}

# Human-readable labels for the UI
TOOL_LABELS: dict[str, str] = {
    "summarize_text": "Summarize",
    "extract_action_items": "Extract Action Items",
    "classify_text": "Classify Document",
    "risk_scan": "Risk Scan",
    "retrieve_context": "Retrieve Context",
}

# Emoji icons for visual clarity
TOOL_ICONS: dict[str, str] = {
    "summarize_text": "📋",
    "extract_action_items": "✅",
    "classify_text": "🏷️",
    "risk_scan": "🚨",
    "retrieve_context": "🔍",
}
