import socket
_old_getaddrinfo = socket.getaddrinfo
def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _old_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _getaddrinfo_ipv4

import os
import re
import json
from dotenv import load_dotenv

load_dotenv()
from typing import Any, Dict, List, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import MODEL_NAME
from agent.graph_rag import (
    build_generation_messages,
    get_llm,
    prepare_for_stream,
    run_agent,
)
from kg.graph_store import get_graph
from security import events
from security.guardrails import (
    REFUSAL,
    detect_injection,
    detect_prompt_leak,
    sanitize_output,
)
from security.ratelimit import chat_limiter
from security.redteam import run_redteam

# 1. Configuration
app = FastAPI(title="Ahmed Portfolio API — Agentic GraphRAG")

# 2. CORS — locked down by default (security showcase). Override with the
#    CORS_ORIGINS env var (comma-separated) in production.
DEFAULT_ORIGINS = [
    "https://bargady.online",
    "https://www.bargady.online",
    "https://front.bargady.online",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
origins = os.getenv("CORS_ORIGINS")
allow_origins = [o.strip() for o in origins.split(",")] if origins else DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # preview deployments
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# 3. Data Models
class ChatRequest(BaseModel):
    question: str


class ImageReference(BaseModel):
    image: str
    alt: str


class ChatResponse(BaseModel):
    answer: str
    images: List[ImageReference] = []
    route: str = ""
    trace: List[Dict[str, Any]] = []
    graph_path: Dict[str, Any] = {}
    security: Dict[str, Any] = {}


class TextRequest(BaseModel):
    text: str


# 4. Helpers
IMAGE_PATTERN = r"!\[(.*?)\]\s*\((.*?)\)"


def _client_id(request: Request) -> str:
    """Best-effort client identity for rate limiting (honours proxy headers)."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def screen_input(question: str, client_id: str) -> Dict[str, Any]:
    """Run pre-generation guardrails. Returns a dict with `allow` plus a
    serializable verdict and a trace step when blocked."""
    if not chat_limiter.check(client_id):
        events.record("rate_limited", risk="low", detail="rate limit exceeded")
        return {"allow": False, "kind": "rate_limited",
                "message": "You're sending messages a bit fast — give me a moment.",
                "verdict": {"blocked": True, "risk": "low",
                            "reason": "Rate limit exceeded.", "signatures": []}}

    verdict = detect_injection(question)
    if verdict.blocked:
        events.record("injection_blocked", risk=verdict.risk,
                      detail=verdict.reason, signatures=verdict.signatures)
        return {"allow": False, "kind": "injection_blocked",
                "message": REFUSAL, "verdict": verdict.to_dict()}

    events.record("clean", risk=verdict.risk)
    return {"allow": True, "verdict": verdict.to_dict()}


def _split_text_and_images(raw: str, fallback_images: List[Dict[str, str]]):
    """Pull markdown images out of the answer; fall back to graph-derived
    images when the model didn't embed any."""
    matches = re.findall(IMAGE_PATTERN, raw)
    images = [ImageReference(image=url.strip(), alt=alt.strip())
              for alt, url in matches]
    if not images and fallback_images:
        images = [ImageReference(**img) for img in fallback_images[:3]]
    clean = re.sub(IMAGE_PATTERN, "", raw).strip()
    return clean, images


# 5. Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request):
    """Run the full agentic GraphRAG pipeline (non-streaming), guardrailed."""
    screen = screen_input(body.question, _client_id(request))
    if not screen["allow"]:
        return ChatResponse(
            answer=screen["message"],
            trace=[{"step": "shield", "detail": screen["kind"]}],
            security=screen["verdict"],
        )
    try:
        state = run_agent(body.question)
        # Output guardrails: block prompt leaks, redact PII.
        safe_answer, info = sanitize_output(state.get("answer", ""))
        if info["leak_blocked"]:
            events.record("leak_blocked", risk="high", detail="system prompt leak")
        if info["pii_redacted"]:
            events.record("pii_redacted", risk="low",
                          detail=",".join(info["pii_redacted"]))

        clean, images = _split_text_and_images(
            safe_answer, state.get("images", [])
        )
        return ChatResponse(
            answer=clean,
            images=images,
            route=state.get("route", ""),
            trace=state.get("trace", []),
            graph_path=state.get("graph_path", {}),
            security=screen["verdict"],
        )
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


async def stream_response(question: str, client_id: str) -> AsyncGenerator[str, None]:
    """Stream the agent's thinking trace, then the answer token-by-token."""
    try:
        import asyncio
        # Pre-generation guardrails (visible security step).
        screen = screen_input(question, client_id)
        if not screen["allow"]:
            yield f"data: {json.dumps({'trace': {'step': 'shield', 'detail': screen['kind']}})}\n\n"
            yield f"data: {json.dumps({'security': screen['verdict']})}\n\n"
            yield f"data: {json.dumps({'chunk': screen['message']})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Run routing + (corrective) retrieval first.
        state = prepare_for_stream(question)

        # Emit the reasoning trace so far.
        for step in state.get("trace", []):
            yield f"data: {json.dumps({'trace': step})}\n\n"
            await asyncio.sleep(0)

        # Emit the traversed subgraph (the UI can highlight it).
        if state.get("graph_path"):
            yield f"data: {json.dumps({'graph_path': state['graph_path']})}\n\n"
            await asyncio.sleep(0)

        # Final reasoning step before tokens start.
        yield f"data: {json.dumps({'trace': {'step': 'answering', 'detail': ''}})}\n\n"
        await asyncio.sleep(0)

        messages = build_generation_messages(state)

        # Stream the final generation using requests stream
        import requests
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        prompt_content = messages[0]["content"] if isinstance(messages[0], dict) else messages[0].content
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        groq_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
            "User-Agent": "Mozilla/5.0",
        }
        groq_payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt_content}],
            "temperature": 0.3,
            "max_tokens": 600,
            "stream": True,
        }

        buf = ""
        in_think = False
        with requests.post(groq_url, headers=groq_headers, json=groq_payload, verify=False, stream=True, timeout=30.0) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=False):
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    text = data["choices"][0]["delta"].get("content", "") or ""
                    if not text:
                        continue

                    # Filter out reasoning/thinking tags from qwen/deepseek models
                    if "<think>" in text:
                        in_think = True
                        continue
                    if "</think>" in text:
                        in_think = False
                        parts = text.split("</think>")
                        text = parts[-1].lstrip()
                        if not text:
                            continue
                    elif in_think:
                        continue

                    buf += text
                    if detect_prompt_leak(buf):
                        events.record("leak_blocked", risk="high",
                                      detail="system prompt leak (stream)")
                        yield f"data: {json.dumps({'trace': {'step': 'shield', 'detail': 'leak_blocked'}})}\n\n"
                        yield f"data: {json.dumps({'chunk': ' …[response withheld: I keep my configuration private.]'})}\n\n"
                        break
                    yield f"data: {json.dumps({'chunk': text})}\n\n"
                    await asyncio.sleep(0)
                except Exception:
                    pass

        prompt_content = messages[0]["content"] if isinstance(messages[0], dict) else messages[0].content
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        groq_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
            "User-Agent": "Mozilla/5.0",
        }
        groq_payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt_content}],
            "temperature": 0.3,
            "max_tokens": 600,
            "stream": True,
        }

        buf = ""
        in_think = False
        with requests.post(groq_url, headers=groq_headers, json=groq_payload, verify=False, stream=True, timeout=30.0) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=False):
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    text = data["choices"][0]["delta"].get("content", "") or ""
                    if not text:
                        continue

                    # Filter out reasoning/thinking tags from qwen/deepseek models
                    if "<think>" in text:
                        in_think = True
                        continue
                    if "</think>" in text:
                        in_think = False
                        parts = text.split("</think>")
                        text = parts[-1].lstrip()
                        if not text:
                            continue
                    elif in_think:
                        continue

                    buf += text
                    if detect_prompt_leak(buf):
                        events.record("leak_blocked", risk="high",
                                      detail="system prompt leak (stream)")
                        yield f"data: {json.dumps({'trace': {'step': 'shield', 'detail': 'leak_blocked'}})}\n\n"
                        yield f"data: {json.dumps({'chunk': ' …[response withheld: I keep my configuration private.]'})}\n\n"
                        break
                    yield f"data: {json.dumps({'chunk': text})}\n\n"
                except Exception:
                    pass

        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        print(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/chat/stream")
async def chat_stream_endpoint(body: ChatRequest, request: Request):
    """Streaming endpoint: reasoning trace + graph path + typed answer."""
    return StreamingResponse(
        stream_response(body.question, _client_id(request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/graph")
def graph_full():
    """The entire knowledge graph — for the frontend's ambient visualization."""
    return get_graph().export_full()


@app.post("/graph/query")
def graph_query(request: ChatRequest):
    """Just the graph retrieval for a question (no LLM). Lets the UI preview
    the subgraph a question would traverse."""
    return get_graph().retrieve(request.question)


@app.get("/security/stats")
def security_stats():
    """Aggregate counters + recent (redacted) security events for the dashboard."""
    return {"stats": events.stats(), "recent": events.recent(40)}


@app.post("/security/screen")
def security_screen(body: TextRequest):
    """Run the injection detector on arbitrary text — powers the free-input
    box of the public red-team playground. (Does not call the LLM.)"""
    return detect_injection(body.text).to_dict()


@app.post("/security/redteam")
def security_redteam():
    """Run the canned attack battery against the live guardrails."""
    return run_redteam()


@app.get("/health")
def health_check():
    """Health check. Graph is always available; LLM/vector are best-effort."""
    kg_ok = True
    try:
        stats = get_graph().stats
    except Exception as e:
        kg_ok = False
        stats = {"error": str(e)}

    return {
        "status": "healthy" if kg_ok else "degraded",
        "model": MODEL_NAME,
        "graph": stats,
        "llm_configured": bool(os.getenv("GROQ_API_KEY")),
        "vector_configured": bool(
            os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX_NAME")
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
