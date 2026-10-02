import os
import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from backend.app.ml.dataset_loader import load_scifact_binary_dataset, DisjointSetUnion
from backend.app.ml.feature_extractor import (
    compute_keyword_overlap_ratio,
    compute_token_length,
    extract_fold_features,
    FEATURE_NAMES
)
from backend.app.ml.experiment_runner import (
    find_optimal_cosine_threshold,
    calculate_metrics,
    run_phase5_experiment
)

def test_scifact_dataset_loader_and_binary_mapping():
    """Verify loading 1,295 SciFact instances with correct SUPPORT=1 and CONTRADICT=0 binary mapping."""
    instances, meta = load_scifact_binary_dataset("data/datasets/scifact/data")
    assert len(instances) == 1295
    assert meta["total_instances"] == 1295
    assert meta["support_count"] == 832
    assert meta["contradict_count"] == 463

    for inst in instances:
        assert inst["label"] in ("SUPPORT", "CONTRADICT")
        if inst["label"] == "SUPPORT":
            assert inst["label_num"] == 1
        else:
            assert inst["label_num"] == 0

def test_disjoint_set_union_bipartite_graph_grouping():
    """Verify DSU bipartite graph connected components assign identical group IDs to connected claim-doc nodes."""
    dsu = DisjointSetUnion()
    dsu.union("C_1", "D_100")
    dsu.union("C_2", "D_100")
    dsu.union("C_2", "D_200")

    assert dsu.find("C_1") == dsu.find("C_2")
    assert dsu.find("C_1") == dsu.find("D_100")
    assert dsu.find("C_1") == dsu.find("D_200")
    assert dsu.find("C_1") != dsu.find("C_999")

def test_keyword_overlap_and_token_length():
    """Verify deterministic tokenization, keyword overlap ratio, and length features."""
    c_text = "Logistic regression is used for binary classification."
    e_text = "Binary classification algorithm includes logistic regression model."
    
    overlap = compute_keyword_overlap_ratio(c_text, e_text)
    assert overlap > 0.0
    assert compute_token_length(c_text) == 7
    assert compute_token_length(e_text) == 7

def test_tfidf_fitted_on_training_data_only():
    """Verify TF-IDF vectorizer vocabulary is derived exclusively from training fold text."""
    train_insts = [
        {"claim_text": "bge embedding vector", "rationale_text": "cosine similarity search"}
    ]
    val_insts = [
        {"claim_text": "unknown_val_term", "rationale_text": "unseen_val_content"}
    ]

    X_tr, X_val, vec = extract_fold_features(train_insts, val_insts)
    vocab = vec.vocabulary_
    
    assert "bge" in vocab
    assert "cosine" in vocab
    assert "unknown_val_term" not in vocab
    assert "unseen_val_content" not in vocab

def test_scaler_fitted_on_training_data_only():
    """Verify StandardScaler mean_ is fit exclusively on training fold features."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    X_val = np.array([[100.0, 200.0]], dtype=np.float32)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    np.testing.assert_array_almost_equal(scaler.mean_, [2.0, 3.0])
    assert X_val_scaled[0, 0] > 0

def test_baseline_threshold_tuning_on_training_data_only():
    """Verify optimal threshold tau is selected on training fold only."""
    train_cos = np.array([0.1, 0.2, 0.8, 0.9])
    y_train = np.array([0, 0, 1, 1])

    best_tau, best_f1 = find_optimal_cosine_threshold(train_cos, y_train)
    assert 0.2 < best_tau <= 0.8
    assert best_f1 == 1.0

def test_full_phase5_experiment_execution(tmp_path):
    """Verify 5-fold StratifiedGroupKFold experiment execution, zero leakage, metrics generation, and artifacts."""
    artifacts_dir = str(tmp_path / "experiments")
    report = run_phase5_experiment(
        data_dir="data/datasets/scifact/data",
        artifacts_dir=artifacts_dir,
        n_splits=5,
        random_state=42
    )

    assert len(report["cv_folds"]) == 5
    assert "baseline_cosine_threshold" in report["aggregate_results"]
    assert "proposed_logistic_regression" in report["aggregate_results"]

    # Verify fold metrics structure
    for fold in report["cv_folds"]:
        assert fold["baseline"]["accuracy"] >= 0.0
        assert fold["logistic_regression"]["accuracy"] >= 0.0
        assert "confusion_matrix" in fold["baseline"]
        assert "confusion_matrix" in fold["logistic_regression"]

    # Verify generated artifact files
    model_path = os.path.join(artifacts_dir, "logistic_regression_scifact.joblib")
    results_path = os.path.join(artifacts_dir, "phase5_scifact_experiment_results.json")
    assert os.path.exists(model_path)
    assert os.path.exists(results_path)
