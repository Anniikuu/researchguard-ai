# System Architecture

ResearchGuard AI is designed with a strict conceptual and architectural boundary between its offline experimental research phase and its online production runtime.

## 1. Controlled Research Experiment (Offline)

The Phase 5 ML experiment runs entirely offline using the SciFact dataset. Its purpose is to evaluate the hypothesis that Logistic Regression outperforms simple cosine similarity for claim verification.

**Experiment Pipeline**:
1. **Dataset Loading**: 1,295 labeled SciFact claim-evidence pairs are loaded.
2. **Leakage Prevention**: Disjoint Set Union groups the dataset into 388 connected components to ensure strict train/val separation during 5-fold cross-validation.
3. **Feature Extraction**: Five features (Cosine similarity via BGE, TF-IDF similarity, keyword overlap, claim length, evidence length) are extracted. TF-IDF and Scaler are fit strictly on training folds.
4. **Baseline**: An optimal cosine threshold is tuned on the training fold and evaluated on the validation fold.
5. **Proposed Model**: Logistic Regression is trained on the scaled features.
6. **Artifact Generation**: The final model, scaler, and vectorizer are trained on the full dataset and saved as a reproducible `.joblib` artifact.

## 2. Runtime Application Pipeline (Online)

The online application provides a local RAG interface for ingesting arbitrary PDFs and verifying generated claims.

**Runtime Pipeline**:
1. **PDF Ingestion**: PyMuPDF extracts text, preserving page numbers.
2. **Chunking**: Text is split into 500-token chunks with 50-token overlap.
3. **Embedding**: `BAAI/bge-small-en-v1.5` creates 384-dimensional vectors.
4. **Storage**: Chunks and vectors are persisted to PostgreSQL utilizing the `pgvector` extension.
5. **Retrieval**: A user query triggers vector cosine similarity search to retrieve the top-K relevant chunks.
6. **RAG Generation**: The local `qwen2.5:7b-instruct` model (via Ollama) generates an answer grounded strictly in the retrieved context.
7. **Claim Extraction**: The LLM extracts testable, atomic scientific claims from its own generated answer.
8. **Evidence Retrieval**: For each claim, a secondary vector search retrieves the single most relevant chunk as evidence.
9. **ML Verification**: The Phase 5 ML artifact (`logistic_regression_scifact.joblib`) transforms the claim-evidence pair into the 5 deterministic features and predicts SUPPORT or CONTRADICT probabilities.
10. **API Response**: The API aggregates the answer, claims, evidence, and verification statuses, returning the payload to the frontend React dashboard.

---
**Critical Distinction**: 
Metrics derived from the SciFact Controlled Research Experiment evaluate the supervised verification model on curated scientific abstracts. They **do not** represent the accuracy of the Runtime Application Pipeline when processing arbitrary out-of-domain PDFs.
