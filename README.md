# Smart Document Action Assistant

A context-augmented, agentic web application that automatically decides how to process any document you give it — email, meeting notes, policy, contract, or question — by routing through an LLM-powered decision layer.

Built for a graduate course on context-augmented GenAI systems.

---

## What makes it agentic

Most document AI apps follow a fixed pipeline: every input gets summarized, or every input gets classified. This app is different.

A dedicated **agent router** — a Claude LLM call with a structured routing prompt — reads the input and dynamically decides which tool to invoke. The app itself has no hard-coded rules. The same input type (e.g. an email) may route to `extract_action_items` one time and `risk_scan` another, depending on the content.

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│  Agent Router  (claude-sonnet-4-6)      │
│  Returns: { selected_tool, confidence,  │
│             rationale, needs_clari? }   │
└────────────────┬────────────────────────┘
                 │  dynamic dispatch
       ┌─────────┼──────────────────────┐
       ▼         ▼          ▼           ▼
  summarize  extract    risk_scan   retrieve
  _text      _action    (tool 3)    _context
             _items                 (tool 5)
  (tool 1)   (tool 2)     │
                       classify
                       _text
                       (tool 4)
```

If confidence is below 0.6 or `needs_clarification` is true, the app pauses and asks the user a follow-up question instead of guessing.

---

## Features

| Feature | Detail |
|---------|--------|
| **5 tools** | Summarize, Extract Action Items, Classify, Risk Scan, Retrieve Context |
| **Agentic routing** | LLM picks the tool, not hard-coded logic |
| **Clarification path** | Low-confidence inputs trigger a follow-up question |
| **File upload** | `.txt` and `.pdf` supported |
| **Observability panel** | Live metrics, request table, detail inspector |
| **SQLite logging** | Full audit trail of every request |
| **User feedback** | Thumbs up / thumbs down stored per request |
| **Evaluation suite** | 15 labeled test cases, accuracy + latency metrics |
| **Sample inputs** | Four pre-loaded examples for quick demo |

---

## Project structure

```
smart-doc-assistant/
├── app.py                          # Main Streamlit page
├── pages/
│   └── 1_Observability.py          # Admin / metrics panel
├── agent/
│   ├── router.py                   # LLM-based agent router (core)
│   └── tools.py                    # 5 tool implementations
├── core/
│   ├── logger.py                   # SQLite observability logger
│   └── metrics.py                  # Routing accuracy + latency metrics
├── evaluation/
│   ├── test_cases.json             # 15 labeled test cases
│   ├── run_evaluation.py           # Runs router over test set
│   ├── compute_metrics.py          # Reads results and prints report
│   └── results/                    # Output of evaluation runs
├── data/                           # SQLite DB lives here (gitignored)
├── .streamlit/config.toml          # Streamlit theme
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd smart-doc-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key from console.anthropic.com |

---

## Running the evaluation

```bash
# Run the router on all 15 test cases
python evaluation/run_evaluation.py

# Print the metrics report from saved results
python evaluation/compute_metrics.py
```

The evaluation reports:
- **Routing accuracy** (quality metric) — % of test cases where the router chose the correct tool
- **Average latency** (operational metric) — end-to-end router call time in ms
- Per-tool breakdown
- Misrouted cases with rationale
- Improvement recommendations

---

## Deployment

### Streamlit Community Cloud (recommended — free)

1. Push this repo to GitHub (no secrets — `.env` is gitignored).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Select your repo, set `app.py` as the entrypoint.
4. In **Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
5. Click **Deploy**.

### Render

1. Create a new **Web Service** from your GitHub repo.
2. Set **Build command**: `pip install -r requirements.txt`
3. Set **Start command**: `streamlit run app.py --server.port $PORT --server.headless true`
4. Add environment variable `ANTHROPIC_API_KEY`.

### Hugging Face Spaces

1. Create a new Space, SDK = **Streamlit**.
2. Push the repo (Spaces reads `requirements.txt` automatically).
3. Add `ANTHROPIC_API_KEY` in the Space's **Secrets** tab.

---

## Architecture notes

### Why Streamlit-only (no FastAPI)?

A single Streamlit app is sufficient for this project. FastAPI would add latency, complexity, and a second process with no benefit for a demo/class project. Streamlit's session state handles the clarification loop cleanly.

### Why SQLite?

Zero infrastructure, ships with Python, sufficient for hundreds of requests. For production, swap in PostgreSQL by changing the connection string in `core/logger.py`.

### Router model choice

The router uses `claude-sonnet-4-6` because it produces reliable structured JSON and understands nuanced document intent. For lower latency, swap to `claude-haiku-4-5-20251001` in `agent/router.py` (routing calls only).

### Agentic vs. non-agentic

This app is agentic because:
1. The LLM has agency over which tool runs — no rule-based dispatch
2. The agent can pause and request more information (clarification path)
3. The routing decision is observable, structured, and reasoned — not a black box

---

## Metrics

| Metric | Type | How measured |
|--------|------|--------------|
| Routing accuracy | Quality | % of labeled test cases with correct tool selection |
| Avg end-to-end latency | Operational | Total ms per request, logged to SQLite |
| Clarification rate | Operational | % of requests that triggered clarification |
| Error rate | Operational | % of requests with a logged error |
| Thumbs-up rate | Quality (user) | % of rated requests marked helpful |

---

## Report section support

| Report section | Where to look |
|---------------|---------------|
| System design / architecture | `agent/router.py` docstring, this README |
| Why it's agentic | `app.py` module docstring, README section above |
| Observability | `core/logger.py`, `pages/1_Observability.py` |
| Metrics | `core/metrics.py`, `evaluation/compute_metrics.py` |
| Evaluation | `evaluation/` folder, `run_evaluation.py` output |
| Deployment | Deployment section above |
| Reflection / failures | `evaluation/results/eval_results.json` after running eval |

---

## License

MIT
