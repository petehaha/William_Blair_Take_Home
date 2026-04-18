# Acquisition Target Screener

A two-stage pipeline that takes a target company profile, scores historical acquirers against it, and produces LLM-generated M&A feasibility assessments as PDFs.

---

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy ollama pydantic jinja2 weasyprint markdown duckduckgo-search
```

**2. Start Ollama and sign in**
```bash
ollama signin
```

**3. Open the notebook**
```bash
jupyter notebook acquisition_screener.ipynb
```

**4. Edit the two input cells and run all**
- **Cell: Target Company Profile** — fill in the target's sector, geography, deal size, etc.
- **Cell: LLM & Pipeline Config** — set `OLLAMA_MODEL` to the cloud model you want, adjust `TOP_N_PIPELINE` if needed.

Then *Run All Cells*. Scored results appear inline; PDF assessments are written to `./assessments/`.

---

## Architecture Overview

```
CSV dataset
    └─► Scoring engine (acquirer profiles + weighted scoring)
            └─► Top N ranked acquirers
                    └─► LLM agent (tool calling + web search)
                            └─► Pydantic-validated JSON
                                    └─► Jinja2 → WeasyPrint → PDF
```

---

## How the CSV Is Used

The dataset (`ma_transactions_500.csv`) contains 500 historical M&A transactions with fields including sector, geography, deal size, EBITDA margin, ownership type, deal type, EV multiples, and outcome.

**Acquirer profiling** — for each unique acquirer, the dataset is aggregated into a statistical profile:
- Sector/geography/ownership frequency distributions (used as fit scores)
- Deal size mean + standard deviations
- EBITDA margin mean + standard deviations
- Last deal year (used for recency decay)

**Qualitative descriptor resolution** — when a user specifies `"strong"` or `"above average"` for EBITDA margin, the system resolves this to the corresponding percentile (p75, p62, etc.) of the *target's sector* within the dataset. This makes descriptors sector-relative rather than absolute.

**LLM context** — each acquirer's historical deals from the dataset are passed verbatim into the LLM prompt as a deals table, providing grounded precedent data for the assessment.

Only **closed** deals are used for acquirer profiling to avoid signal contamination from withdrawn or pending transactions.

---

## Scoring Engine

Six weighted dimensions (weights configurable in the profile cell) gleaned both from the instructions and research:

| Dimension | Default Weight | Method |
|---|---|---|
| Sector fit | 30% | Share of acquirer's past deals in target sector |
| Deal size fit | 25% | Z-score proximity to acquirer's historical mean |
| Geography fit | 15% | Share of deals in target geography (National/Multi-Regional count as 0.5×) |
| EBITDA margin fit | 15% | Z-score proximity to acquirer's historical margin |
| Ownership fit | 10% | Share of deals with same ownership type |
| Recency | 5% | Exponential decay: `100 × 0.8^(2025 − last_deal_year)` |

Composite score is the weighted sum (0–100). Grades: A ≥ 80, B ≥ 65, C ≥ 50, D < 50.

---

## LLM Prompt Architecture

### Model
Ollama (`gpt-oss:20b-cloud` by default). Requires a model with tool-calling support. `temperature=0`, `seed=42` for deterministic output.

**Model selection rationale:** SOTA models (`gemma4:31b`) produce higher quality output but exceed the ~1 minute end-to-end target for the full acquirer pipeline. `gpt-oss:20b-cloud` balances quality and speed at this scale.

### Two-phase design

**Phase 1 — Research (tool calling)**
The system prompt instructs the model to issue all web searches as a single parallel batch before writing anything. The `web_search` tool uses the `duckduckgo-search` DDGS library (`ddgs.text()`), which returns real web results rather than the DDG instant answer API (which only returns entity/knowledge panel data and is largely useless for company-specific queries). Tool calls within a round execute in parallel via `ThreadPoolExecutor`.

**Phase 2 — Assessment (structured output)**
After research, the model returns a single JSON object with exactly eight keys. Chain-of-thought reasoning is captured in the `reasoning_trace` field within the JSON itself — no separate thinking mode is used. The system prompt also describes the PDF rendering pipeline (Markdown → WeasyPrint, A4 one-page, two-column layout for sections 3 & 4) so the model constrains its output length and formatting accordingly.

### Retry loop
Phase 1 (research + assessment) runs once. On JSON parse or Pydantic validation failure, Phase 2 retries are JSON-fix only — a lightweight `chat()` call without tools, passing the schema and error back to the model. Web searches are never re-run on retry. Up to `MAX_RETRIES` attempts.

### Parallelism
All N acquirer agents run concurrently in a `ThreadPoolExecutor`, staggered by a configurable delay (`stagger_secs`, default 2s) to avoid overwhelming the Ollama server with simultaneous requests (429s). Each agent has its own independent conversation thread. Tool calls within each agent round also run in parallel.

---

## Assumptions

**About the dataset**
- Acquirers with only 1–2 deals in the dataset have high uncertainty; their profiles use a ±40% size range as a std proxy
- "National" and "Multi-Regional" geography entries are treated as partial matches (0.5×) for any specific region
- Deal year is used as a proxy for acquirer activity recency; the dataset spans 2016–2024

**About the target profile**
- `deal_size_mm` is the expected transaction value, not revenue
- Qualitative descriptors (`"strong"`, etc.) are resolved against the *sector distribution in the dataset*, not real-world benchmarks
- Ownership type reflects the target's current status, not post-transaction structure

**About the LLM**
- The model has general knowledge of named acquirers (e.g. Cerner, Epic) that supplements the web search results
- Web search uses `DDGS.text()` — real web results, but quality varies by acquirer name recognition
- `temperature=0` + `seed=42` produces near-deterministic outputs but is not fully guaranteed across model versions
- Chain-of-thought is elicited via the `reasoning_trace` JSON field rather than a native thinking mode

---

## Known Limitations & Failure Modes

| Issue | Impact | Notes |
|---|---|---|
| Thin acquirer history | Low-confidence scores for acquirers with 1–2 dataset deals | No confidence weighting applied to composite score yet |
| Web search coverage | Some acquirers return sparse results from DDGS | Affects Company Overview and recent news quality; less-known acquirers hit hardest |
| JSON validation failures | Assessment skipped after `MAX_RETRIES` exhausted | Smaller models fail more frequently |
| Sector label mismatch | Target sector not found in dataset → falls back to full dataset for percentile resolution | User must use exact sector names from the dataset |
| Recency bias | Acquirers with no deals since 2020 score ≤ 33 on recency regardless of fit | By design, but may penalise legitimate strategic buyers |
| LLM Rate Limiting | Ollama rate limits users, so parallelizing the workflow can cause errors despite staggered API calls | In paid models, this rate limit would also certainly be increased, reducing the latency and error rate. |

---

## Future Work

1. **Screener hyperparameter tuning** — the scoring weights, z-score thresholds, and recency decay constant were set heuristically. A labelled ground-truth dataset of known acquirer-target pairs would allow proper optimisation (e.g. grid search or Bayesian tuning) to find weights that better predict real outcomes.

2. **Reranker stage** — inserting a lightweight reranker between the statistical screener and the LLM pipeline would improve signal quality. The screener is fast but coarse; a reranker trained on deal-level features (e.g. EV multiples, deal type, strategic rationale tags) could refine the top-N before the expensive LLM calls, reducing wasted inference on weak candidates.

3. **Model speed/quality tradeoff** — significant time was spent finding a model that completes the full pipeline in under ~1 minute. SOTA 70B+ models produce noticeably better assessments but are too slow at this concurrency level. A dedicated benchmarking sweep across model sizes (7B, 14B, 20B, 32B) on this specific task — measuring both output quality and wall-clock time — would yield a more principled selection.

4. **PDF layout and formatting** — the current one-page template is functional but basic. Improvements would include a proper cover header with branding, better typography, conditional page breaks for longer sections, and potentially a multi-page layout with an appendix for the full precedent transaction table. Some of the rendering could be improved as well.

5. **Richer tool calls with dedicated M&A data sources** — the current `web_search` tool is general-purpose. Replacing or supplementing it with dedicated integrations would substantially improve assessment quality: PitchBook/Mergermarket APIs for verified deal data, SEC EDGAR for public filing context, and company websites for stated strategic priorities. Each could be a separate tool the model selects from based on what it needs.

---
