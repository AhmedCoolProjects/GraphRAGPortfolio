# 🧠 GraphRAG Digital Twin & Interactive Portfolio

> **Production Monorepo**: Next.js 15 (React 19, Turbopack, Tailwind CSS v4, Motion) + FastAPI Agentic GraphRAG Backend (LangGraph, Knowledge Graph, Pinecone Vector Search, LLM Guardrails).

Created by **Ahmed Bargady** — PhD Student in AI & Cybersecurity at **UM6P (Mohammed VI Polytechnic University)**.

---

<!-- PHOTO PLACEHOLDER: Main Banner / Hero Screenshot -->
<!-- <img src="./docs/images/hero-banner.png" alt="GraphRAG Digital Twin Banner" width="100%" /> -->

---

## 🌟 Architecture Overview

This monorepo powers an AI-driven digital twin designed to answer complex technical, research, and career inquiries in real-time. Unlike standard RAG systems, it combines **Agentic Corrective RAG (LangGraph)**, structured **Knowledge Graphs**, and **AI Security Guardrails**.

<!-- PHOTO PLACEHOLDER: Architecture Diagram / System Overview -->
<!-- <img src="./docs/images/architecture-diagram.png" alt="GraphRAG Architecture Diagram" width="100%" /> -->

```text
               ┌────────────────────────────────────────────────────────┐
               │              Next.js 15 Frontend (App Router)          │
               │   Chat Interface • Knowledge Graph Viz • Security Hub   │
               └───────────────────────────┬────────────────────────────┘
                                           │ SSE / Stream
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │             FastAPI GraphRAG Server Endpoint           │
               └───────────────────────────┬────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
           ┌────────────────────────┐            ┌────────────────────────┐
           │   AI Security Shield   │            │   LangGraph State Engine│
           │ Prompt Leak & Injection│            │  Adaptive Routing Node │
           └────────────────────────┘            └───────────┬────────────┘
                                                             │
                                        ┌────────────────────┴────────────────────┐
                                        ▼                                         ▼
                           ┌────────────────────────┐                ┌────────────────────────┐
                           │   Knowledge Graph      │                │   Pinecone Vector DB   │
                           │ Multi-Hop Entity Facts │                │  Semantic Document RAG │
                           └────────────────────────┘                └────────────────────────┘
```

---

## ⚡ Core Technical Highlights

### 1. 🤖 Agentic GraphRAG (LangGraph)
- **Adaptive Routing**: Automatically detects query intent (*Smalltalk*, *General Knowledge*, *Entity-Grounded*, or *Multi-hop Complex*).
- **Corrective Retrieval Loop**: Grades retrieved facts; if context is insufficient, it rewrites the query dynamically and re-retrieves before generation.
- **Hybrid Retrieval**: Fuses deterministic Knowledge Graph subgraphs with high-dimensional vector embeddings for max recall and precision.

<!-- PHOTO PLACEHOLDER: GraphRAG Thinking Trace & Reasoning UI -->
<!-- <img src="./docs/images/graphrag-thinking-trace.png" alt="GraphRAG Reasoning Trace" width="100%" /> -->

---

### 2. 🕸 Interactive Knowledge Graph
- **Entity Linking**: Maps research domains (APT Detection, Provenance Graphs, GNNs, Transformers) to papers, projects, and skills.
- **Live Traversal Stream**: Emits active graph paths alongside response streams, rendering interactive node topologies in the UI.

<!-- PHOTO PLACEHOLDER: Knowledge Graph Topology View -->
<!-- <img src="./docs/images/knowledge-graph-viz.png" alt="Knowledge Graph Topology Visualization" width="100%" /> -->

---

### 3. 🛡 AI Security Showcase & Red-Teaming Guardrails
- **Prompt Injection Defense**: Pre-execution input classification blocking adversarial jailbreaks.
- **System Prompt Leak Guard**: Output stream monitoring to prevent sensitive context disclosure.
- **Rate Limiting & Red-Teaming Suite**: Built-in endpoints for automated security evaluation.

<!-- PHOTO PLACEHOLDER: Security Showcase & Red-Teaming Dashboard -->
<!-- <img src="./docs/images/security-showcase.png" alt="Security Guardrails & Red Teaming" width="100%" /> -->

---

## 📁 Repository Layout

```text
GraphRAGPortfolio/
├── frontend/             # Next.js 15 (React 19, Turbopack, Tailwind CSS v4, Motion, Lucide)
│   ├── src/
│   │   ├── app/          # App router pages (Home, Chat, Security Showcase)
│   │   ├── components/   # Interactive Graph, Chat UI, Reasoning Trace, Terminal components
│   │   └── store/        # Zustand state management
│   └── package.json
│
├── backend/              # FastAPI Python Backend
│   ├── agent/            # LangGraph state machine & Corrective RAG pipeline
│   ├── kg/               # Knowledge graph schema, networkx, & store
│   ├── security/         # Red-teaming scripts, guardrails, & rate limiters
│   ├── data/             # Research documents, papers, & bio knowledge base
│   ├── config.py         # Centralized model & server configuration
│   └── requirements.txt
│
├── package.json          # Root npm workspace & script orchestration
├── turbo.json            # Turborepo task pipeline configuration
├── .gitignore
└── .env.example
```

---

## 🚀 Quick Start

### Prerequisites
- **Node.js**: `v20+`
- **npm** or **pnpm**
- **Python**: `v3.10+`

---

### 1. Installation

Clone the repository and install root workspace dependencies:

```bash
git clone https://github.com/your-username/GraphRAGPortfolio.git
cd GraphRAGPortfolio

# Install Node dependencies for all workspaces
npm install
```

Set up Python virtual environment for the backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

---

### 2. Environment Configuration

Copy `.env.example` to set up environment variables:

```bash
cp .env.example .env
```

Configure your environment variables:
- `GROQ_API_KEY`: Groq API Key for fast LLM inference.
- `PINECONE_API_KEY`: Pinecone Vector Store API Key.
- `NEXT_PUBLIC_CHAT_API_URL`: Backend URL (e.g. `http://localhost:8000`).

---

### 3. Local Development

Run both Frontend and Backend concurrently with Turborepo:

```bash
npm run dev
```

- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`

---

## 📜 License

MIT License © [Ahmed Bargady](https://github.com/your-username)
