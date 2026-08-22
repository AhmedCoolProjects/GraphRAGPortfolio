# 🧠 GraphRAG Digital Twin & Interactive Portfolio

> **Production Monorepo**: Next.js 15 (React 19, Turbopack, Tailwind CSS v4, Motion) + FastAPI Agentic GraphRAG Backend (LangGraph, Knowledge Graph, Pinecone Vector Search, LLM Guardrails).

Created by **Ahmed Bargady** — PhD Student in AI & Cybersecurity at **UM6P (Mohammed VI Polytechnic University)**.

---

<img width="1728" height="996" alt="image" src="https://github.com/user-attachments/assets/8e348d85-224c-4ef8-8a76-9cefeb62d7ac" />


---

## 🌟 Architecture Overview

This monorepo powers an AI-driven digital twin designed to answer complex technical, research, and career inquiries in real-time. Unlike standard RAG systems, it combines **Agentic Corrective RAG (LangGraph)**, structured **Knowledge Graphs**, and **AI Security Guardrails**.



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

<img width="1728" height="996" alt="image" src="https://github.com/user-attachments/assets/a12c7aaf-1666-4bbf-bbfd-2596810fe412" />


---

### 2. 🕸 Interactive Knowledge Graph
- **Entity Linking**: Maps research domains (APT Detection, Provenance Graphs, GNNs, Transformers) to papers, projects, and skills.
- **Live Traversal Stream**: Emits active graph paths alongside response streams, rendering interactive node topologies in the UI.

<img width="1728" height="996" alt="image" src="https://github.com/user-attachments/assets/8d22319e-dd59-444d-a537-4c283b00ad2c" />


---

### 3. 🛡 AI Security Showcase & Red-Teaming Guardrails
- **Prompt Injection Defense**: Pre-execution input classification blocking adversarial jailbreaks.
- **System Prompt Leak Guard**: Output stream monitoring to prevent sensitive context disclosure.
- **Rate Limiting & Red-Teaming Suite**: Built-in endpoints for automated security evaluation.

<img width="1728" height="996" alt="image" src="https://github.com/user-attachments/assets/4732f7f5-b50f-4eae-bde7-9266f3f029d5" />


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
git clone https://github.com/AhmedCoolProjects/GraphRAGPortfolio.git
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

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.

---

## 👨‍💻 Connect & Follow

<div align="center">

### **Ahmed Bargady**
*PhD Student in AI & Cybersecurity @ UM6P*

[![Portfolio](https://img.shields.io/badge/Website-bargady.online-000000?style=for-the-badge&logo=react&logoColor=61DAFB)](https://bargady.online/)
[![GitHub](https://img.shields.io/badge/GitHub-AhmedCoolProjects-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AhmedCoolProjects)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ahmed_Bargady-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-bargady/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ahmedBargady-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ahmedBargady)
[![X / Twitter](https://img.shields.io/badge/X-@AhmedBargady-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/AhmedBargady)

</div>

