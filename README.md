# ResearchGuard AI

ResearchGuard AI is a Retrieval-Augmented Generation (RAG) system with a claim-level verification step using a supervised Logistic Regression classifier. 
It is designed as an AI/ML research project to answer the core research question:

**"Can a supervised Logistic Regression classifier improve claim verification compared with a simple semantic-similarity threshold in a RAG-based question-answering system?"**

## Setup (Local Only)

This project strictly relies on local AI models via Ollama and Hugging Face. No external cloud APIs are used.

### Prerequisites
- Docker & Docker Compose (for PostgreSQL + pgvector and Ollama)
- Python 3.10+
- Node.js 18+

### Quick Start

1. Start infrastructure:
   ```bash
   docker-compose up -d
   ```

2. Start Backend:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\Activate.ps1
   pip install -r backend/requirements.txt
   uvicorn backend.app.main:app --reload
   ```

3. Start Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
