# Phase 5 Experiment Results

This document presents the official results of the Phase 5 Controlled Research Experiment comparing a Cosine Similarity baseline against a supervised Logistic Regression classifier for binary claim verification.

## Experimental Setup

- **Dataset**: SciFact (1,295 labeled claim-evidence instances: 832 SUPPORT, 463 CONTRADICT).
- **Features**: 
  1. Cosine similarity
  2. TF-IDF similarity
  3. Keyword overlap ratio
  4. Claim length
  5. Evidence length
- **Cross-Validation**: 5-fold Stratified Group K-Fold (leakage-safe via bipartite connected-component grouping).
- **Baseline**: Cosine similarity threshold, tuned exclusively on training folds to maximize F1 score.
- **Proposed Model**: Scikit-Learn Logistic Regression with `class_weight="balanced"`.

## Results (Mean ± Std over 5 folds)

| Metric | Baseline (Cosine Threshold) | Proposed (Logistic Regression) |
| :--- | :--- | :--- |
| **Accuracy** | 0.6456 ± 0.0050 | 0.5764 ± 0.0490 |
| **Precision** | 0.6468 ± 0.0048 | 0.6892 ± 0.0398 |
| **Recall** | 0.9880 ± 0.0114 | 0.6204 ± 0.0654 |
| **F1 Score** | 0.7817 ± 0.0041 | 0.6517 ± 0.0484 |
| **ROC-AUC** | 0.5945 ± 0.0572 | 0.5948 ± 0.0460 |

## Interpretation

The experimental results demonstrate trade-offs between the two approaches:

- **Logistic Regression achieved higher mean Precision** (0.6892 vs 0.6468), indicating that when it predicted SUPPORT, it was more likely to be correct than the baseline. It identified substantially more CONTRADICT instances than the cosine threshold baseline.
- **The cosine baseline achieved higher mean Accuracy, Recall, and F1**. However, its high Recall and F1 occurred alongside a very high tendency to predict SUPPORT (as evidenced by the confusion matrices where the baseline rarely predicted CONTRADICT).
- **ROC-AUC was approximately comparable** (0.5945 for baseline vs 0.5948 for Logistic Regression), suggesting similar overall discriminatory power across varying thresholds.

These results do not establish a universal superiority of either approach. The optimal choice depends on whether the system priorities favor minimizing false positives (favoring Logistic Regression) or maximizing the capture of true positive evidence (favoring the Cosine Baseline).

*(Note: No statistical significance test has been established for these results.)*
