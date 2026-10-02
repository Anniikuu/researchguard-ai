import re
import math
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from ..services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "cosine_similarity",
    "tfidf_similarity",
    "keyword_overlap_ratio",
    "claim_length",
    "evidence_length",
]

def tokenize_text(text: str) -> List[str]:
    """Simple deterministic tokenization for keyword overlap and word length."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())

def compute_keyword_overlap_ratio(claim_text: str, evidence_text: str) -> float:
    """
    Computes Jaccard keyword overlap ratio: |Claim_tokens ∩ Evidence_tokens| / |Claim_tokens ∪ Evidence_tokens|.
    """
    claim_tokens = set(tokenize_text(claim_text))
    evidence_tokens = set(tokenize_text(evidence_text))
    if not claim_tokens or not evidence_tokens:
        return 0.0
    intersection = claim_tokens.intersection(evidence_tokens)
    union = claim_tokens.union(evidence_tokens)
    return len(intersection) / len(union)

def compute_token_length(text: str) -> int:
    """Computes word token count of text."""
    return len(tokenize_text(text))

_EMBEDDING_CACHE: Dict[str, List[float]] = {}

def get_cached_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Retrieves or computes normalized BGE embeddings for a list of texts using an in-memory cache."""
    uncached = [t for t in texts if t not in _EMBEDDING_CACHE]
    if uncached:
        unique_uncached = list(set(uncached))
        new_vecs = embedding_service.generate_embeddings_batch(unique_uncached)
        for t, vec in zip(unique_uncached, new_vecs):
            _EMBEDDING_CACHE[t] = vec

    return [_EMBEDDING_CACHE[t] for t in texts]

def extract_bge_cosine_similarities(claim_texts: List[str], evidence_texts: List[str]) -> np.ndarray:
    """
    Batch generates BAAI/bge-small-en-v1.5 embeddings for claim and evidence texts
    and computes pairwise cosine similarities. Uses embedding cache for efficiency.
    """
    if not claim_texts or not evidence_texts or len(claim_texts) != len(evidence_texts):
        return np.zeros(len(claim_texts), dtype=np.float32)

    claim_embeddings = np.array(get_cached_embeddings_batch(claim_texts), dtype=np.float32)
    evidence_embeddings = np.array(get_cached_embeddings_batch(evidence_texts), dtype=np.float32)

    # Dot product of normalized vectors equals cosine similarity
    cosine_sims = np.sum(claim_embeddings * evidence_embeddings, axis=1)
    return np.clip(cosine_sims, -1.0, 1.0)


def extract_fold_features(
    train_instances: List[Dict[str, Any]],
    val_instances: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Extracts the 5 features for training and validation splits following strict leakage-free rules:
    - BGE cosine similarity (pretrained model, no fitting).
    - TF-IDF similarity (TfidfVectorizer fit ONLY on training texts).
    - Deterministic keyword overlap ratio.
    - Claim token length.
    - Evidence token length.
    """
    # 1. Fit TF-IDF Vectorizer ONLY on training fold text
    train_corpus = [inst["claim_text"] for inst in train_instances] + [inst["rationale_text"] for inst in train_instances]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)
    vectorizer.fit(train_corpus)

    def _extract_instance_features(instances: List[Dict[str, Any]], bge_cosines: np.ndarray) -> np.ndarray:
        features_list = []
        claim_texts = [inst["claim_text"] for inst in instances]
        rationale_texts = [inst["rationale_text"] for inst in instances]

        # TF-IDF transform
        claim_tfidf = vectorizer.transform(claim_texts)
        rationale_tfidf = vectorizer.transform(rationale_texts)

        # Pairwise TF-IDF cosine similarity
        tfidf_sims = np.array([
            float(sklearn_cosine(claim_tfidf[i], rationale_tfidf[i])[0, 0])
            for i in range(len(instances))
        ], dtype=np.float32)

        for i, inst in enumerate(instances):
            kw_overlap = compute_keyword_overlap_ratio(inst["claim_text"], inst["rationale_text"])
            c_len = float(compute_token_length(inst["claim_text"]))
            e_len = float(compute_token_length(inst["rationale_text"]))
            cos_sim = float(bge_cosines[i])

            features_list.append([
                cos_sim,
                float(tfidf_sims[i]),
                kw_overlap,
                c_len,
                e_len
            ])

        return np.array(features_list, dtype=np.float32)

    # BGE Cosine similarities
    train_claims = [inst["claim_text"] for inst in train_instances]
    train_evidences = [inst["rationale_text"] for inst in train_instances]
    val_claims = [inst["claim_text"] for inst in val_instances]
    val_evidences = [inst["rationale_text"] for inst in val_instances]

    train_bge_cos = extract_bge_cosine_similarities(train_claims, train_evidences)
    val_bge_cos = extract_bge_cosine_similarities(val_claims, val_evidences)

    X_train = _extract_instance_features(train_instances, train_bge_cos)
    X_val = _extract_instance_features(val_instances, val_bge_cos)

    return X_train, X_val, vectorizer
