# GraphRAG Portfolio Monorepo

Welcome to the **GraphRAG Portfolio** repository! This monorepo combines an interactive, modern Next.js frontend with an AI-powered FastAPI backend utilizing Knowledge Graphs and Vector RAG (Retrieval-Augmented Generation).

---

## 🏗 Repository Structure

```text
GraphRAGPortfolio/
├── frontend/             # Next.js 15+ (React 19, Tailwind CSS v4, Motion, Shadcn UI)
│   ├── src/              # App routes, interactive chat, knowledge graph visualization
│   └── package.json      # Frontend UI dependencies
│
├── backend/              # FastAPI Python Backend (GraphRAG, LangGraph, Vector Store)
│   ├── main.py           # FastAPI entrypoint & endpoints
│   ├── agent/            # LangGraph / Graph RAG agent pipeline
│   ├── kg/               # Knowledge graph schema & store
│   ├── security/         # Security, guardrails, & rate limiting
│   ├── data/             # Markdown docs & biography knowledge base
│   ├── requirements.txt  # Python package dependencies
│   └── package.json      # Workspace runner script for Python
│
├── package.json          # Root Monorepo configuration (Workspaces + Turborepo)
├── turbo.json            # Turborepo task pipeline configuration
├── .gitignore            # Git ignore definitions for Node & Python
└── .env.example          # Environment variables template
```

---

## ⚡ Features

- **Interactive AI Assistant**: GraphRAG agent answering technical, personal, and research questions.
- **Knowledge Graph Visualization**: Interactive graph views linking topics, publications, and skills.
- **Security & Guardrails**: Built-in prompt injection defense, rate limiting, and output moderation.
- **Turborepo Orchestration**: Fast, unified monorepo task execution (`dev`, `build`, `lint`).

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

# Install Node dependencies across workspace
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

### 2. Environment Setup

Copy `.env.example` to your environment files:

```bash
cp .env.example .env
```

Fill in your required API keys (e.g. `GROQ_API_KEY`, `PINECONE_API_KEY`, `NEXT_PUBLIC_API_URL`).

---

### 3. Running Development Mode

From the root directory, run both Frontend and Backend concurrently with Turborepo:

```bash
npm run dev
```

Or run individual services:

- **Frontend only** (`http://localhost:3000`):
  ```bash
  npm run dev:frontend
  ```

- **Backend only** (`http://localhost:8000`):
  ```bash
  npm run dev:backend
  ```

---

## 📜 License

MIT License © Ahmed Bargady
