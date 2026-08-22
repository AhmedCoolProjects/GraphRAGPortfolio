# Digital Twin — Agentic GraphRAG Backend

The twin that answers visitors on [bargady.online](https://bargady.online) is no
longer a plain RAG chatbot. It runs an **agentic GraphRAG** pipeline: a
hand-curated knowledge graph of my career, retrieved by a LangGraph agent that
routes, retrieves, self-corrects, and answers in my voice — and streams its own
reasoning to the UI.

## Pipeline

```
route ──┬─ smalltalk / general ───────────────► generate
        ├─ grounded (clear entity) ─► retrieve ─► generate
        └─ complex (multi-hop)     ─► retrieve ─► grade ─┐
                                        ▲                │ insufficient
                                        └── rewrite ◄────┘  (corrective RAG)
```

1. **Adaptive routing** (`route_node`) — query complexity is matched to pipeline
   complexity. "Hi" or "what is a transformer?" skip retrieval entirely; a
   question naming an entity takes the grounded path; a multi-hop, relational
   question ("what did you build that combines security and ML?") takes the
   corrective path. Routing is structural and deterministic — fast and free.

2. **Hybrid retrieval** (`retrieve_node`) — the **knowledge graph** is the
   primary store: the question is entity-linked to seed nodes, then expanded into
   a focused *k*-hop subgraph (the person hub is anchored but never traversed
   through, which keeps each subgraph tight). The existing **Pinecone** vector
   store is queried in parallel when configured, for prose nuance. Graph facts +
   vector chunks compose the context.

3. **Corrective reflection** (`grade_node` → `rewrite_node`) — on the complex
   path, an LLM grades whether the retrieved context can actually answer the
   question. If not, it rewrites the query and retrieves again (bounded). This is
   the CRAG / Self-RAG idea applied to a personal graph.

4. **Generation** (`generate_node`) — answers in first person from
   `TEMPLATE_ME.md`, streamed token-by-token.

## What the frontend gets

The streaming endpoint emits more than tokens:

| Event | Purpose |
|-------|---------|
| `{"trace": {step, detail}}` | the twin's visible "thinking" steps |
| `{"graph_path": {nodes, edges}}` | the exact subgraph traversed — for the live graph visualization |
| `{"chunk": "..."}` | answer tokens (typing effect) |
| `{"done": true}` | end of stream |

## The graph

`kg/knowledge_graph.json` — 80 nodes / 120 edges, hand-curated and validated
against the typed schema in `kg/schema.py`. Node types: person, institution,
company, role, project, skill, research area, domain, book, language, hobby.

- `kg/graph_store.py` — NetworkX loader, entity linking, *k*-hop retrieval,
  fact serialization, and full-graph export.
- `GET /graph` — the whole graph (for the ambient visualization).
- `POST /graph/query` — the subgraph a question would traverse, no LLM.

## Run locally

```bash
pip install -r requirements.txt
python scripts/test_graph.py     # graph + retrieval smoke test (no API keys)
uvicorn main:app --reload        # needs GROQ_API_KEY; Pinecone optional
```

The system **degrades gracefully**: with no Pinecone it runs graph-only; the
graph layer needs no API keys at all.

## Security layer (`security/`)

Every message is screened before it reaches the model, and every answer is
screened before it leaves. All of this is deterministic and key-free, so it's
fully unit-testable (`scripts/test_security.py`).

- **Prompt-injection / jailbreak detection** (`guardrails.py`) — categorized,
  weighted signatures (instruction override, system-prompt extraction, role
  hijack, exfiltration, encoding/delimiter evasion) produce a risk score and an
  explainable verdict. Medium/high risk is blocked before any LLM call.
- **Rate limiting** (`ratelimit.py`) — in-memory token bucket per client.
- **Output guardrails** (`guardrails.py`) — a system-prompt-leak guard (blocks
  the model regurgitating its own instructions, including a rolling check during
  streaming) and **PII redaction** (emails, phones, IPs, cards, API keys).
- **Observability** (`events.py`) — a bounded, PII-free event log + aggregate
  stats behind `GET /security/stats`.
- **Public red-team demo** (`redteam.py`) — a battery of known attacks run
  against the live guardrails (`POST /security/redteam`), plus `POST
  /security/screen` for arbitrary input. Powers the `/security` playground page.
- CORS is locked to known origins by default (`CORS_ORIGINS` to override),
  replacing the previous `allow_origins=["*"]`.
- The graph is a closed, validated vocabulary — the twin can only ground answers
  in vetted facts about me, which constrains hallucination.

When an input is blocked, the chat stream emits a `shield` trace step and a
`security` event, so the defense is visible in the UI.
