# ResearchGuard AI

ResearchGuard AI is a Retrieval-Augmented Generation (RAG) system with a claim-level verification step using a supervised Logistic Regression classifier. It is designed as a fully reproducible AI/ML research project.

## Research Question
**"Can a supervised Logistic Regression classifier improve evidence-based claim verification compared with a simple semantic-similarity threshold in a RAG-based question-answering system?"**

## Core Architecture & Features
1. **PDF Ingestion**: PyMuPDF-based text extraction with page-aware chunking.
2. **Retrieval**: Vector cosine similarity search via PostgreSQL (`pgvector`) using `BAAI/bge-small-en-v1.5` embeddings.
3. **RAG Workflow**: Local generation and scientific claim extraction using `qwen2.5:7b-instruct` (via Ollama).
4. **Claim Verification**: Supervised Scikit-Learn Logistic Regression model evaluating claim-evidence pairs on 5 distinct features.
5. **UI Dashboard**: React-based frontend for document management and query verification tracking.

## Tech Stack
- **Backend**: FastAPI, Python 3.10+, SQLAlchemy, Pydantic, Scikit-Learn, PyMuPDF.
- **Frontend**: React 19, TypeScript, Vite, TailwindCSS.
- **Infrastructure**: PostgreSQL + `pgvector`, Ollama (Local LLMs).
- **Machine Learning**: `BAAI/bge-small-en-v1.5`, `qwen2.5:7b-instruct`, Logistic Regression.

---

## Installation & Setup

This project strictly relies on local AI models. No external cloud APIs are used.

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+

### 1. Start Infrastructure (PostgreSQL & Ollama)
```bash
docker-compose up -d
```
*Note: Ensure Ollama is running and the model is pulled (`ollama pull qwen2.5:7b-instruct`). PostgreSQL migrations will run automatically on backend startup.*

### 2. Phase 5 Experiment Reproduction
Before the runtime API can verify claims, the ML verification artifact must be generated locally. (The artifact is excluded from Git).
See [docs/reproducibility.md](docs/reproducibility.md) for full dataset download instructions.
Once the dataset is placed in `data/datasets/scifact/data/`, run:
```bash
python -c "from backend.app.ml.experiment_runner import run_phase5_experiment; run_phase5_experiment()"
```

### 3. Start Backend
```bash
python -m venv .venv
# Activate virtual environment (e.g., .venv/bin/activate or .venv\Scripts\Activate.ps1)
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

### 4. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Testing
The backend is fully tested utilizing `pytest`. To run tests:
```bash
pytest
```
*Tests cover the health endpoint, PDF processing, API flows, RAG integration, claim extraction, and ML artifact loading.*

## Project Documentation
Detailed documentation is provided in the `docs/` directory:
- [Architecture](docs/architecture.md)
- [Reproducibility Guide](docs/reproducibility.md)
- [Dataset Provenance](docs/dataset.md)
- [Model Provenance](docs/models.md)
- [Experiment Results](docs/results.md)
- [Research Integrity](docs/research_integrity.md)
- [Limitations](docs/limitations.md)
- [Research Paper Package](docs/paper_package.md)

### Reproducibility Notes
We mandate strict data hygiene. The `.env` files, `pgdata/`, virtual environments, and generated `.joblib` model artifacts are intentionally excluded from Git. The experiment uses fixed random seeds and programmatic leakage assertions to ensure verifiable results.

### Limitations Summary
Metrics derived from the SciFact Controlled Experiment evaluate the model on curated scientific abstracts and do not directly translate to accuracy when verifying claims against arbitrary runtime PDFs. See [docs/limitations.md](docs/limitations.md) for a comprehensive discussion on domain mismatch, class imbalance, and LLM dependency constraints.
