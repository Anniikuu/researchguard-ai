import os
import json
import logging
import joblib
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from .dataset_loader import load_scifact_binary_dataset
from .feature_extractor import extract_fold_features, FEATURE_NAMES

logger = logging.getLogger(__name__)

def find_optimal_cosine_threshold(train_cosines: np.ndarray, y_train: np.ndarray) -> Tuple[float, float]:
    """
    Learns optimal cosine similarity threshold on training fold ONLY by maximizing F1 score.
    Returns (best_threshold, best_train_f1).
    """
    best_tau = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.0, 1.0, 101)

    for tau in thresholds:
        preds = (train_cosines >= tau).astype(int)
        score = float(f1_score(y_train, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)

    return best_tau, best_f1

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculates binary classification evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()  # [[TN, FP], [FN, TP]]

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm  # Rows: True [0=CONTRADICT, 1=SUPPORT], Cols: Predicted
    }

def run_phase5_experiment(
    data_dir: str = "data/datasets/scifact/data",
    artifacts_dir: str = "data/experiments",
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes Phase 5 Controlled Research Experiment:
    1. Loads SciFact binary dataset (1,295 SUPPORT/CONTRADICT pairs).
    2. Builds bipartite connected components for leakage-safe grouping.
    3. Runs 5-fold StratifiedGroupKFold cross validation.
    4. Programmatically asserts zero group, document, claim, and rationale leakage on every fold.
    5. Evaluates Cosine Baseline (train-only threshold tuning) vs Logistic Regression on identical validation instances.
    6. Produces aggregate metrics (mean ± std) and exports reproducible experiment artifacts.
    """
    logger.info("Starting Phase 5 Controlled Research Experiment...")
    instances, dataset_meta = load_scifact_binary_dataset(data_dir)

    X_indices = np.arange(len(instances))
    y_all = np.array([inst["label_num"] for inst in instances], dtype=int)
    groups = [inst["group_id"] for inst in instances]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    baseline_fold_metrics = []
    lr_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_indices, y_all, groups=groups), 1):
        logger.info(f"--- Running CV Fold {fold_idx}/{n_splits} ---")
        train_insts = [instances[i] for i in train_idx]
        val_insts = [instances[i] for i in val_idx]

        # --- PROGRAMMATIC ZERO-LEAKAGE ASSERTIONS ---
        train_groups = set(groups[i] for i in train_idx)
        val_groups = set(groups[i] for i in val_idx)
        assert len(train_groups.intersection(val_groups)) == 0, f"Fold {fold_idx}: Group leakage detected!"

        train_docs = set(inst["doc_id"] for inst in train_insts)
        val_docs = set(inst["doc_id"] for inst in val_insts)
        assert len(train_docs.intersection(val_docs)) == 0, f"Fold {fold_idx}: Document ID leakage detected!"

        train_claims = set(inst["claim_id"] for inst in train_insts)
        val_claims = set(inst["claim_id"] for inst in val_insts)
        assert len(train_claims.intersection(val_claims)) == 0, f"Fold {fold_idx}: Claim ID leakage detected!"

        train_claim_texts = set(inst["claim_text"].lower() for inst in train_insts)
        val_claim_texts = set(inst["claim_text"].lower() for inst in val_insts)
        assert len(train_claim_texts.intersection(val_claim_texts)) == 0, f"Fold {fold_idx}: Claim text leakage detected!"

        train_rationales = set(inst["rationale_text"] for inst in train_insts)
        val_rationales = set(inst["rationale_text"] for inst in val_insts)
        assert len(train_rationales.intersection(val_rationales)) == 0, f"Fold {fold_idx}: Rationale text leakage detected!"

        train_tuples = set((inst["claim_text"].lower(), inst["doc_id"], tuple(inst["sentences"]), inst["label"]) for inst in train_insts)
        val_tuples = set((inst["claim_text"].lower(), inst["doc_id"], tuple(inst["sentences"]), inst["label"]) for inst in val_insts)
        assert len(train_tuples.intersection(val_tuples)) == 0, f"Fold {fold_idx}: Exact tuple leakage detected!"

        # Extract features (TF-IDF fit ONLY on training fold)
        X_train, X_val, fold_vectorizer = extract_fold_features(train_insts, val_insts)
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        # 1. BASELINE: Cosine Similarity Threshold (tuned on TRAIN ONLY)
        train_cosines = X_train[:, 0]
        val_cosines = X_val[:, 0]
        best_tau, best_train_f1 = find_optimal_cosine_threshold(train_cosines, y_train)

        val_baseline_preds = (val_cosines >= best_tau).astype(int)
        baseline_metrics = calculate_metrics(y_val, val_baseline_preds, val_cosines)
        baseline_metrics["selected_threshold"] = best_tau

        # 2. PROPOSED: Logistic Regression (Scaler & Model fit on TRAIN ONLY)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        val_lr_preds = clf.predict(X_val_scaled)
        val_lr_probs = clf.predict_proba(X_val_scaled)[:, 1]
        lr_metrics = calculate_metrics(y_val, val_lr_preds, val_lr_probs)

        baseline_fold_metrics.append(baseline_metrics)
        lr_fold_metrics.append(lr_metrics)

        fold_summary = {
            "fold": fold_idx,
            "train_size": len(train_insts),
            "val_size": len(val_insts),
            "train_support": int(np.sum(y_train == 1)),
            "train_contradict": int(np.sum(y_train == 0)),
            "val_support": int(np.sum(y_val == 1)),
            "val_contradict": int(np.sum(y_val == 0)),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "baseline": baseline_metrics,
            "logistic_regression": lr_metrics,
        }
        fold_results.append(fold_summary)

    def _aggregate_metric(fold_list: List[Dict[str, Any]], key: str) -> Tuple[float, float]:
        vals = [f[key] for f in fold_list]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    baseline_agg = {}
    lr_agg = {}

    for k in metrics_keys:
        b_mean, b_std = _aggregate_metric(baseline_fold_metrics, k)
        l_mean, l_std = _aggregate_metric(lr_fold_metrics, k)
        baseline_agg[k] = {"mean": b_mean, "std": b_std}
        lr_agg[k] = {"mean": l_mean, "std": l_std}

    # --- TRAIN FINAL DEPLOYABLE MODEL ARTIFACT ON FULL DATASET ---
    logger.info("Training final deployable model artifact on all 1,295 SciFact dataset instances...")
    X_full, _, final_vectorizer = extract_fold_features(instances, instances)
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)

    final_clf = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
    final_clf.fit(X_full_scaled, y_all)

    # Prepare export directory
    os.makedirs(artifacts_dir, exist_ok=True)

    model_artifact_path = os.path.join(artifacts_dir, "logistic_regression_scifact.joblib")
    model_artifact = {
        "model": final_clf,
        "scaler": final_scaler,
        "vectorizer": final_vectorizer,
        "feature_names": FEATURE_NAMES,
        "dataset_size": len(instances),
        "random_state": random_state,
    }
    joblib.dump(model_artifact, model_artifact_path)

    results_json_path = os.path.join(artifacts_dir, "phase5_scifact_experiment_results.json")
    experiment_report = {
        "experiment": "Phase 5 Controlled Research Experiment: Cosine Baseline vs Logistic Regression",
        "research_question": "Can supervised Logistic Regression improve evidence-based claim verification compared with cosine similarity alone?",
        "dataset_metadata": dataset_meta,
        "features": FEATURE_NAMES,
        "cv_folds": fold_results,
        "aggregate_results": {
            "baseline_cosine_threshold": baseline_agg,
            "proposed_logistic_regression": lr_agg,
        },
        "artifacts": {
            "model_path": model_artifact_path,
            "results_path": results_json_path,
        }
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    logger.info(f"Phase 5 experiment completed successfully. Results saved to {results_json_path}")
    return experiment_report
