# Model Provenance

ResearchGuard AI utilizes three primary machine learning models. This document outlines their purpose, architecture, and configuration within the repository.

## 1. Embedding Model: BAAI/bge-small-en-v1.5

- **Purpose**: Generates semantic vector representations for page-aware PDF chunks, claims, and evidence texts to enable cosine similarity computations and vector retrieval.
- **Dimensions**: 384
- **Loading Mechanism**: Loaded via `SentenceTransformer` (Hugging Face) in `EmbeddingService`.
- **Pretrained Status**: Used entirely off-the-shelf (pretrained). The model is not retrained or fine-tuned during the Phase 5 experiment or runtime.
- **Relevant Configuration**: 
  - Defined in `config.py` as `EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"`.
  - Vectors are normalized (`normalize_embeddings=True`) prior to output.

## 2. Large Language Model (LLM): qwen2.5:7b-instruct

- **Purpose**: Performs grounded RAG generation, answering user questions based on retrieved PDF context, and extracts testable scientific claims from the generated answer.
- **Provider / Backend**: Served locally via Ollama.
- **Pretrained Status**: Used off-the-shelf. No fine-tuning or LoRA adaptation is performed.
- **Relevant Configuration**:
  - Model Name: `qwen2.5:7b-instruct`
  - Context Window (`OLLAMA_NUM_CTX`): `4096`
  - Temperature (`OLLAMA_TEMPERATURE`): `0.0` (Ensures deterministic, highly grounded generation).
- **Local Requirements**: Requires the Ollama daemon running locally with the `qwen2.5:7b-instruct` model pulled (`ollama pull qwen2.5:7b-instruct`).

## 3. Claim Verification Classifier: Logistic Regression

- **Purpose**: A supervised classifier that evaluates 5 explicit features (cosine similarity, TF-IDF similarity, keyword overlap, claim length, evidence length) to classify a claim-evidence pair as SUPPORT (1) or CONTRADICT (0).
- **Architecture**: Scikit-Learn `LogisticRegression`.
- **Pretrained Status**: **Trained in ResearchGuard** during the Phase 5 Controlled Experiment using the SciFact dataset.
- **Relevant Configuration**:
  - `class_weight`: `"balanced"` (Adjusts for the 64/36 class imbalance in SciFact).
  - `random_state`: `42`
  - `max_iter`: `1000`
- **Artifact Requirements**: The trained model, along with its associated `StandardScaler` and `TfidfVectorizer` (vocabulary max_features=5000, ngram_range=(1,2)), is serialized via joblib to `data/experiments/logistic_regression_scifact.joblib`. 
- **Loading Mechanism**: Loaded by `MLInferenceService` at runtime for Phase 6 claim verification. *Note: As this artifact is generated, it is ignored by Git and must be reproduced locally.*
