"""
Agent Router — the core agentic component.

This module is the ONLY place that decides what happens next.
It uses an LLM to read the user's input and dynamically select
which tool to invoke, rather than following a fixed pipeline.

Agentic behavior: given the same type of document (e.g. an email),
the router may choose different tools depending on the content signals —
tasks vs. risks vs. informational content vs. ambiguity.
"""
import json
import time
from typing import Union, Optional

import anthropic
from pydantic import BaseModel, field_validator

from core.client import get_model_id

# ── Routing prompt ────────────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """You are an intelligent document-routing agent.
Your ONLY job is to read a piece of text and decide which single processing
tool is most appropriate. You do NOT process the text yourself.

Available tools:
  summarize_text        — Long or informational content that needs condensing
                          (reports, articles, briefings, FYI emails, announcements)
  extract_action_items  — Content that contains tasks, requests, deadlines,
                          or next steps (meeting notes, project emails, to-dos)
  classify_text         — When the primary need is to identify / categorize the
                          document (ambiguous genre, multiple categories, audit)
  risk_scan             — Content with legal, financial, compliance, safety, or
                          reputational red flags (contracts, policies, incidents)
  retrieve_context      — Input is a question or references concepts that need
                          background knowledge to handle correctly

Routing signals:
  • "action", "please do", "by Friday", deadlines, owner names → extract_action_items
  • "FYI", "for your records", long explanatory prose            → summarize_text
  • "what type", "what is this", genre ambiguity                 → classify_text
  • "must not", liability, GDPR, compliance, "risk"              → risk_scan
  • "what is", "explain", "how does" (questions)                 → retrieve_context
  • Very short (<30 words), no clear intent                      → low confidence + clarification

Return ONLY valid JSON — no markdown fences, no extra text:
{
  "selected_tool": "<tool_name>",
  "confidence": <float 0.0–1.0>,
  "needs_clarification": <boolean>,
  "rationale": "<1–2 sentences explaining your choice>",
  "clarification_question": "<question for the user if needs_clarification is true, else null>"
}"""

VALID_TOOLS = {
    "summarize_text",
    "extract_action_items",
    "classify_text",
    "risk_scan",
    "retrieve_context",
}


class RouterOutput(BaseModel):
    selected_tool: str
    confidence: float
    needs_clarification: bool
    rationale: str
    clarification_question: Optional[str] = None

    @field_validator("selected_tool")
    @classmethod
    def validate_tool(cls, v: str) -> str:
        if v not in VALID_TOOLS:
            raise ValueError(f"Unknown tool: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


def route_input(
    text: str,
    client: Union[anthropic.Anthropic, anthropic.AnthropicBedrock],
) -> tuple[RouterOutput, float]:
    """
    Core agentic routing step.

    Sends the user's input to the LLM and parses a structured routing decision.
    Returns (RouterOutput, router_latency_seconds).

    The router is intentionally isolated: it only decides, it never executes.
    Execution happens in tools.py, called from app.py after this returns.
    """
    start = time.perf_counter()

    message = client.messages.create(
        model=get_model_id(),
        max_tokens=512,
        system=ROUTER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                # Cap at 3000 chars so routing stays fast and cheap
                "content": f"Analyze this input and choose the best tool:\n\n{text[:3000]}",
            }
        ],
    )

    latency = time.perf_counter() - start
    raw = message.content[0].text.strip()

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
        output = RouterOutput(**data)
    except Exception:
        # Fallback: route to classify with low confidence so the user sees uncertainty
        output = RouterOutput(
            selected_tool="classify_text",
            confidence=0.4,
            needs_clarification=True,
            rationale="Router could not parse a clean decision from the LLM response.",
            clarification_question="Could you describe what kind of help you need with this text?",
        )

    return output, latency
