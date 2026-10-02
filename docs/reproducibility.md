# Phase 5 Reproducibility Guide

This document outlines the exact process required to reproduce the Phase 5 Controlled Research Experiment comparing a Cosine Similarity baseline to a supervised Logistic Regression classifier on the SciFact dataset.

## 1. Python Environment & Requirements

ResearchGuard AI requires Python 3.10+.
First, set up a virtual environment and install the required backend dependencies:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

## 2. External Dependencies

- **PostgreSQL & pgvector**: Not strictly required for the offline SciFact experiment execution (which runs entirely in memory), but required for the overall application runtime.
- **Ollama**: Not required for the Phase 5 experiment.

## 3. SciFact Dataset Acquisition

The SciFact dataset is explicitly excluded from the Git repository to maintain hygiene. It must be manually downloaded and placed in the appropriate directory before running the experiment.

### Download Process
The SciFact dataset files can be obtained from the official AllenAI SciFact repository (or equivalent source).
You need the following three JSONL files:
1. `corpus.jsonl`
2. `claims_train.jsonl`
3. `claims_dev.jsonl`

### Expected Directory Structure
The dataset loader expects these files to be located at `data/datasets/scifact/data/` relative to the project root.

```
data/
└── datasets/
    └── scifact/
        └── data/
            ├── corpus.jsonl
            ├── claims_train.jsonl
            └── claims_dev.jsonl
```

## 4. Experiment Execution

Once the environment is active and the dataset is in place, you can execute the Phase 5 experiment programmatically.

### Exact Entry Point
The experiment is executed by running the `experiment_runner.py` module as a script, or by importing its main function.

### Exact Command

To run the experiment from the project root:

```bash
python -c "from backend.app.ml.experiment_runner import run_phase5_experiment; run_phase5_experiment()"
```

## 5. Experiment Configuration

The experiment is strictly controlled via `backend/app/ml/experiment_runner.py` with the following configuration:

- **Random Seed**: `42` (Fixed for cross-validation and Logistic Regression initialization).
- **Cross-Validation Configuration**: 5-fold Stratified Group K-Fold (`StratifiedGroupKFold(n_splits=5)`).
- **Leakage Prevention**: Disjoint Set Union (Union-Find) is used to create Bipartite Graph connected components between Claims and Documents. Group IDs ensure no claim or document appears in both train and validation splits simultaneously.
- **Feature Configuration**: 
  1. `cosine_similarity` (BGE-small-en-v1.5)
  2. `tfidf_similarity` (TfidfVectorizer fitted on **train folds only**, max_features=5000, ngram_range=(1,2))
  3. `keyword_overlap_ratio` (Jaccard similarity)
  4. `claim_length` (word token count)
  5. `evidence_length` (word token count)
- **Model Configuration**: `LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)`

## 6. Output Artifacts

Upon successful completion, the experiment generates two artifacts in the `data/experiments/` directory:

1. **`phase5_scifact_experiment_results.json`**
   - **Purpose**: Stores the exact fold-by-fold metrics and aggregated results.
   - **Verification**: You can verify your results by comparing this file's output to the metrics reported in `docs/results.md`.

2. **`logistic_regression_scifact.joblib`**
   - **Purpose**: The deployable model artifact containing the trained `LogisticRegression` model, `StandardScaler`, and `TfidfVectorizer` (all trained on the full 1,295-instance dataset).
   - **Verification**: The runtime inference pipeline (`backend/app/ml/inference.py`) explicitly checks for this file. If successfully generated, the UI dashboard and Phase 6 API will load it automatically.

> **Note**: The `.joblib` file is ignored by `.gitignore` to prevent tracking large binaries. It must be generated locally via the above command before the ML-enhanced runtime will function.
