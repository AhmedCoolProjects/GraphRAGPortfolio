"""
Agentic GraphRAG pipeline for the digital twin.

A real LangGraph state machine:

    route ──┬─ (smalltalk / offtopic) ─────────────► generate
            ├─ (general: no graph hit, technical) ──► generate
            ├─ (grounded: clear entity)  ─► retrieve ─► generate
            └─ (complex: multi-hop)      ─► retrieve ─► grade ─┐
                                              ▲                │ insufficient
                                              └── rewrite ◄────┘  (corrective RAG)

Design notes:
  * Routing is structural + deterministic (fast, free, testable). It decides how
    much machinery a question deserves — simple questions skip the heavy path.
  * Retrieval is HYBRID: the knowledge graph (always) plus the existing Pinecone
    vector store (when configured). Graph gives structured, multi-hop facts;
    vectors give prose nuance.
  * The "complex" path runs a corrective loop: an LLM grades whether retrieved
    facts can answer the question and, if not, rewrites the query and re-retrieves
    (bounded by MAX_RETRIES). This is the genuinely agentic part.
  * Every node appends to `trace`, which the API streams to the UI as the
    twin's visible "thinking", and exposes `graph_path` for the graph viz.

LangGraph and the LLM/vector clients are imported lazily so this module (and the
graph-only logic) stays importable without those deps or API keys present.
"""
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional

from config import MODEL_NAME
from kg.graph_store import get_graph

MAX_RETRIES = 1                      # corrective re-retrievals on the complex path
GRAPH_HOPS = 2

# Patterns that mean "this isn't really about Ahmed's work" -> light path.
_SMALLTALK = re.compile(
    r"\b(hi|hey|hello|how are you|what'?s up|good (morning|evening)|"
    r"joke|poem|sing|weather|your name|who made you|are you (an? )?(ai|bot|robot))\b",
    re.I,
)
# Words that signal a multi-hop / relational question -> corrective path.
_MULTIHOP = re.compile(
    r"\b(combine|combines|combining|both|relationship|relate|related|compare|"
    r"comparison|versus|vs|connect|connection|across|together|intersection|"
    r"overlap|link between|how does .* (relate|connect|fit))\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Lazy LLM / vector clients
# ---------------------------------------------------------------------------
_llm = None
_retriever = "uninitialized"  # sentinel; becomes a retriever or None


def call_groq_sync(prompt_text: str, max_tokens: int = 600) -> str:
    """Direct lightweight Groq call via requests (bypasses socket/urllib issues on Vercel)."""
    import requests
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    res = requests.post(url, headers=headers, json=payload, timeout=30.0)
    res.raise_for_status()
    data = res.json()
    raw = data["choices"][0]["message"]["content"]
    # Strip thinking tags if present from qwen/deepseek models
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    return raw


def get_llm():
    """Wrapper returning a compatibility object with .invoke()."""
    class GroqWrapper:
        def invoke(self, messages):
            text = messages[0]["content"] if isinstance(messages[0], dict) else getattr(messages[0], "content", str(messages[0]))
            class Response:
                def __init__(self, content):
                    self.content = content
            return Response(call_groq_sync(text))
    return GroqWrapper()


def get_vector_retriever():
    """Lazily connect the Pinecone retriever. Returns None if not configured
    so the system degrades gracefully to graph-only retrieval."""
    global _retriever
    if _retriever != "uninitialized":
        return _retriever

    try:
        hf_key = os.getenv("HF_API_KEY")
        pc_key = os.getenv("PINECONE_API_KEY")
        index = os.getenv("PINECONE_INDEX_NAME")
        if not (hf_key and pc_key and index and os.getenv("ENABLE_PINECONE") == "true"):
            _retriever = None
            return None

        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        Pinecone(api_key=pc_key, environment=os.getenv("PINECONE_ENVIRONMENT"))
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=hf_key,
        )
        store = PineconeVectorStore.from_existing_index(
            index_name=index, embedding=embeddings
        )
        _retriever = store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:  # never let vector setup break the twin
        print(f"[graph_rag] vector retriever disabled: {e}")
        _retriever = None
    return _retriever


DEFAULT_PERSONA = """You are Ahmed Bargady answering questions on your personal portfolio site, bargady.online.
You ARE Ahmed. You are not an assistant or chatbot. Speak in first person as a PhD Student in AI and Cybersecurity at UM6P.
CONTEXT:
{context}

QUESTION:
{question}
"""

def load_persona_template() -> str:
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "TEMPLATE_ME.md")
        with open(os.path.abspath(path), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return DEFAULT_PERSONA


# ---------------------------------------------------------------------------
# Node functions (pure-ish; each takes & returns a state dict)
# ---------------------------------------------------------------------------
def _trace(state: Dict[str, Any], step: str, detail: str = "") -> None:
    state.setdefault("trace", []).append({"step": step, "detail": detail})


def route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Adaptive routing: match query complexity to pipeline complexity."""
    q = state["question"]
    seeds = get_graph().link_entities(q)
    state["seeds"] = seeds

    if _SMALLTALK.search(q) and not seeds:
        state["route"] = "smalltalk"
    elif not seeds:
        # No graph hit: a general/technical question (e.g. "what is a transformer?")
        state["route"] = "general"
    elif _MULTIHOP.search(q) or len({s.split(":", 1)[0] for s in seeds}) >= 3:
        state["route"] = "complex"
    else:
        state["route"] = "grounded"

    _trace(state, "routing", f"route={state['route']}, seeds={len(seeds)}")
    return state


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid retrieval: knowledge graph (+ optional vector store)."""
    q = state.get("rewritten_question") or state["question"]
    g = get_graph().retrieve(q, hops=GRAPH_HOPS)
    state["graph_facts"] = g["facts"]
    state["graph_path"] = g["graph_path"]
    state["images"] = g["images"]
    _trace(state, "graph_search",
           f"{len(g['graph_path']['nodes'])} nodes, {len(g['facts'])} facts")

    vector_chunks: List[str] = []
    retriever = get_vector_retriever()
    if retriever is not None:
        try:
            docs = retriever.invoke(q)
            vector_chunks = [d.page_content for d in docs]
            _trace(state, "vector_search", f"{len(vector_chunks)} chunks")
        except Exception as e:
            _trace(state, "vector_search", f"skipped ({e})")

    state["context"] = _compose_context(state["graph_facts"], vector_chunks)
    return state


def grade_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Corrective RAG: does the retrieved context actually answer the question?"""
    state["attempts"] = state.get("attempts", 0) + 1
    facts = state.get("graph_facts", [])
    if not facts and not state.get("context", "").strip():
        state["sufficient"] = False
    else:
        try:
            verdict = _llm_grade(state["question"], state["context"])
            state["sufficient"] = verdict
        except Exception:
            state["sufficient"] = True  # fail open: better to answer than stall
    _trace(state, "verifying",
           f"sufficient={state['sufficient']} (attempt {state['attempts']})")
    return state


def rewrite_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite the query for a better second retrieval pass."""
    try:
        state["rewritten_question"] = _llm_rewrite(
            state["question"], state.get("context", "")
        )
    except Exception:
        state["rewritten_question"] = state["question"]
    _trace(state, "rewriting", state.get("rewritten_question", ""))
    return state


def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final answer generation in Ahmed's voice."""
    messages = build_generation_messages(state)
    llm = get_llm()
    state["answer"] = llm.invoke(messages).content
    _trace(state, "answering", "")
    return state


# ---------------------------------------------------------------------------
# Generation prompt (shared by the graph and the streaming endpoint)
# ---------------------------------------------------------------------------
def build_generation_messages(state: Dict[str, Any]):
    """Return a ChatPromptValue-compatible message list for the LLM."""
    template = load_persona_template()
    context = state.get("context", "").strip()
    if not context:
        context = "(No specific stored memory matched. Answer from your own "
        context += "knowledge as a Ph.D. student, staying in character.)"
    filled = template.replace("{context}", context).replace(
        "{question}", state["question"]
    )
    return [{"role": "user", "content": filled}]


def _compose_context(graph_facts: List[str], vector_chunks: List[str]) -> str:
    parts: List[str] = []
    if graph_facts:
        parts.append("Knowledge-graph facts:\n" + "\n".join(
            f"- {f}" for f in graph_facts
        ))
    if vector_chunks:
        parts.append("Related notes:\n" + "\n\n".join(vector_chunks))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Small LLM helpers for the corrective loop
# ---------------------------------------------------------------------------
def _llm_grade(question: str, context: str) -> bool:
    prompt = (
        "You are a retrieval grader. Decide if the CONTEXT contains enough "
        "information to answer the QUESTION. Reply with exactly 'YES' or 'NO'.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context[:2000]}\n\nAnswer:"
    )
    out = call_groq_sync(prompt, max_tokens=20)
    return out.strip().upper().startswith("Y")


def _llm_rewrite(question: str, context: str) -> str:
    prompt = (
        "The retrieved context was insufficient. Rewrite the user's question "
        "into a single, more retrievable search query about Ahmed Bargady's "
        "career, projects, skills, or research. Return only the rewritten query.\n\n"
        f"Original question: {question}\nRewritten query:"
    )
    out = call_groq_sync(prompt, max_tokens=100)
    return out.strip().strip('"')


# ---------------------------------------------------------------------------
# LangGraph assembly
# ---------------------------------------------------------------------------
def _route_decision(state: Dict[str, Any]) -> str:
    return {
        "smalltalk": "generate",
        "general": "generate",
        "grounded": "retrieve",
        "complex": "retrieve",
    }[state["route"]]


def _grade_decision(state: Dict[str, Any]) -> str:
    if state.get("sufficient"):
        return "generate"
    if state.get("attempts", 0) > MAX_RETRIES:
        return "generate"
    return "rewrite"


def _after_retrieve(state: Dict[str, Any]) -> str:
    # Only the complex path pays for the corrective grade/rewrite loop.
    return "grade" if state["route"] == "complex" else "generate"


_compiled = None


def get_agent():
    """Compile (once) and return the LangGraph app."""
    global _compiled
    if _compiled is not None:
        return _compiled

    from langgraph.graph import END, StateGraph

    workflow = StateGraph(dict)
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("route")
    workflow.add_conditional_edges("route", _route_decision,
                                   {"retrieve": "retrieve", "generate": "generate"})
    workflow.add_conditional_edges("retrieve", _after_retrieve,
                                   {"grade": "grade", "generate": "generate"})
    workflow.add_conditional_edges("grade", _grade_decision,
                                   {"rewrite": "rewrite", "generate": "generate"})
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    _compiled = workflow.compile()
    return _compiled


def run_agent(question: str) -> Dict[str, Any]:
    """Non-streaming convenience wrapper used by /chat."""
    state: Dict[str, Any] = {"question": question}
    return get_agent().invoke(state)


# ---------------------------------------------------------------------------
# Streaming path: run everything up to generation, then stream tokens.
# Reuses the very same node functions so behaviour can't drift.
# ---------------------------------------------------------------------------
def prepare_for_stream(question: str) -> Dict[str, Any]:
    """Execute route -> retrieve -> (grade -> rewrite -> retrieve)* and return
    the final state, WITHOUT generating. The caller streams the answer."""
    state: Dict[str, Any] = {"question": question}
    route_node(state)

    if state["route"] in ("smalltalk", "general"):
        return state

    retrieve_node(state)
    if state["route"] == "complex":
        grade_node(state)
        while not state.get("sufficient") and state.get("attempts", 0) <= MAX_RETRIES:
            rewrite_node(state)
            retrieve_node(state)
            grade_node(state)
    return state
