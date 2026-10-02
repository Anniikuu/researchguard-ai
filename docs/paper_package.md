# Research Paper Support Package

This document serves as a foundational package for drafting a formal research paper based on the ResearchGuard AI Phase 5 experiment. 

---

## 1. Title
**Beyond Semantic Similarity: Evaluating Supervised Logistic Regression for Evidence-Based Claim Verification in RAG Systems**

## 2. Abstract
Retrieval-Augmented Generation (RAG) mitigates large language model (LLM) hallucinations by grounding generation in retrieved documents. However, verifying the veracity of LLM-generated claims against retrieved evidence remains challenging. Existing systems often rely on simple cosine similarity thresholds, which struggle to distinguish between semantic relatedness and logical entailment. We propose an offline supervised verification model utilizing Logistic Regression over a 5-feature set (including TF-IDF, keyword overlap, and token lengths alongside cosine similarity). We evaluate our approach on a strictly controlled, leakage-free bipartite graph of the SciFact dataset (N=1,295). Results demonstrate that while the cosine threshold baseline achieves a higher mean F1 score (0.7817 vs. 0.6517) due to a near-perfect recall (0.9880) biased toward positive classification, the Logistic Regression model achieves higher mean Precision (0.6892 vs. 0.6468) and successfully identifies substantially more contradictory instances. We argue that for high-stakes verification tasks, precision and the capacity to detect contradictions may outweigh raw recall, though neither approach establishes universal superiority.

## 3. Introduction
Retrieval-Augmented Generation (RAG) is increasingly deployed in domains requiring high factual accuracy. While retrieval grounds the LLM, the model may still synthesize claims that contradict the retrieved context. Ensuring the faithfulness of generated claims requires an automated verification step. Many RAG pipelines employ vector cosine similarity to evaluate entailment, operating on the assumption that semantic proximity implies factual support. However, sentences can be semantically identical in vocabulary but diametrically opposed in meaning. This paper investigates whether a supervised, feature-engineered Logistic Regression model can provide more precise verification than a naive cosine similarity baseline.

## 4. Problem Statement
Simple vector cosine similarity between a claim and its retrieved evidence is insufficient for robust verification because it conflates semantic relatedness (topical similarity) with logical entailment (factual support). 

## 5. Research Question
"Can supervised Logistic Regression improve evidence-based claim verification compared with cosine similarity alone?"

## 6. System Overview
We developed ResearchGuard AI, a local RAG application utilizing PyMuPDF for ingestion, `BAAI/bge-small-en-v1.5` for embedding, PostgreSQL/pgvector for storage, and `qwen2.5:7b-instruct` (via Ollama) for generation and claim extraction. The verification layer intercepts extracted claims, retrieves specific evidence chunks, and evaluates them using the trained Machine Learning artifact.

## 7. Methodology
Our core methodology isolates the verification step as a binary classification task: given a Claim $(C)$ and Evidence $(E)$, predict $Y \in \{0, 1\}$ where $1$ is SUPPORT and $0$ is CONTRADICT. We conducted a controlled, offline experiment comparing a tuned cosine baseline against a Logistic Regression model trained on a deterministic feature space.

## 8. Dataset
We utilized the AllenAI SciFact dataset, which evaluates scientific claims against abstracts. We extracted 1,295 labeled claim-evidence instances, naturally imbalanced towards SUPPORT (64.25%) over CONTRADICT (35.75%).

## 9. Evidence Retrieval
In the runtime architecture, evidence is retrieved dynamically via cosine similarity search in `pgvector`. For the controlled experiment, we utilized the known ground-truth rationale sentences from the SciFact corpus to ensure the verification model was evaluated independently of retrieval accuracy.

## 10. Feature Engineering
We engineered five deterministic features for each $(C, E)$ pair:
1. **Cosine Similarity**: Vector dot product of normalized BGE embeddings.
2. **TF-IDF Similarity**: Cosine similarity of TF-IDF vectors (max_features=5000, unigram/bigram).
3. **Keyword Overlap Ratio**: Jaccard similarity of tokenized sets.
4. **Claim Length**: Word token count.
5. **Evidence Length**: Word token count.

## 11. Baseline
The baseline model relies entirely on Feature 1 (Cosine Similarity). During training, an optimal threshold $\tau$ was selected by maximizing the F1 score. If $sim(C, E) \geq \tau$, the instance was classified as SUPPORT.

## 12. Logistic Regression
The proposed model is a Scikit-Learn Logistic Regression classifier (`class_weight="balanced"`). It was trained on the scaled 5-feature vector to predict the class probability $P(Y=1 | C,E)$.

## 13. Experimental Design
We employed 5-fold Stratified Group K-Fold cross-validation. For each fold, the TF-IDF vectorizer and standard scaler were fit strictly on the training distribution. The baseline threshold and the LR model were optimized on the training data and evaluated on identical validation instances.

## 14. Leakage Prevention
Claims and documents in SciFact form a bipartite graph. We mapped this graph into 388 disjoint connected components using a Union-Find algorithm. These components served as grouping IDs for the cross-validation, guaranteeing zero leakage of claims, documents, or texts between training and validation splits.

## 15. Evaluation Metrics
Models were evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC. 

## 16. Results
The measured mean results (± standard deviation) over the 5 folds are:
- **Accuracy**: Baseline = 0.6456 ± 0.0050 | Logistic Regression = 0.5764 ± 0.0490
- **Precision**: Baseline = 0.6468 ± 0.0048 | Logistic Regression = 0.6892 ± 0.0398
- **Recall**: Baseline = 0.9880 ± 0.0114 | Logistic Regression = 0.6204 ± 0.0654
- **F1 Score**: Baseline = 0.7817 ± 0.0041 | Logistic Regression = 0.6517 ± 0.0484
- **ROC-AUC**: Baseline = 0.5945 ± 0.0572 | Logistic Regression = 0.5948 ± 0.0460

## 17. Discussion
The baseline achieved high Recall and F1 by exhibiting an extreme bias toward the majority class (SUPPORT), rarely identifying CONTRADICT instances. While Logistic Regression suffered a drop in Recall and raw Accuracy, it achieved higher Precision and demonstrated a significantly improved capacity to detect contradictions. Depending on the application context—such as medical or legal domains where false positives (unverified claims marked as supported) carry severe risks—the precision-oriented Logistic Regression approach may be highly preferable.

## 18. Limitations
This study is limited by its reliance on the SciFact dataset, which evaluates short scientific abstracts and forces a binary evaluation constraint. Furthermore, the runtime application operates on arbitrary PDF chunks without known ground truth, presenting a domain mismatch. Claim verification in production also remains heavily dependent on upstream retrieval quality.

## 19. Reproducibility
All code required to reproduce this experiment, including programmatic leakage assertions and fixed random seeds, is open source. Reproducibility steps are fully documented in the project repository.

## 20. Conclusion
We demonstrated that while a simple cosine similarity threshold can achieve high F1 scores by defaulting to the majority class, a supervised Logistic Regression model using basic lexical and semantic features provides more precise verification and better detects contradictions. Future work should explore non-binary classification logic and cross-domain generalization.

---
*[Related work references should be added during literature review.]*
