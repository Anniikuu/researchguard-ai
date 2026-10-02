import os
import logging
import numpy as np
import joblib
from typing import Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from .feature_extractor import (
    FEATURE_NAMES,
    tokenize_text,
    compute_keyword_overlap_ratio,
    compute_token_length,
    extract_bge_cosine_similarities
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = os.path.join("data", "experiments", "logistic_regression_scifact.joblib")

class MLInferenceService:
    def __init__(self, artifact_path: Optional[str] = None):
        self.artifact_path = artifact_path or DEFAULT_ARTIFACT_PATH
        self._artifact: Optional[Dict[str, Any]] = None
        self.model = None
        self.scaler = None
        self.vectorizer = None

    def load_artifact(self) -> None:
        """Loads and validates the Phase 5 deployable ML artifact."""
        if self._artifact is not None:
            return

        if not os.path.exists(self.artifact_path):
            # Try looking relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            alt_path = os.path.join(project_root, self.artifact_path)
            if os.path.exists(alt_path):
                self.artifact_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Phase 5 ML model artifact not found at '{self.artifact_path}' or '{alt_path}'."
                )

        logger.info(f"Loading Phase 5 ML artifact from: {self.artifact_path}")
        artifact = joblib.load(self.artifact_path)

        # Validate required artifact keys
        required_keys = ["model", "scaler", "vectorizer", "feature_names"]
        for key in required_keys:
            if key not in artifact:
                raise KeyError(f"ML artifact missing required key: '{key}'")

        # Validate feature ordering
        artifact_features = artifact["feature_names"]
        if artifact_features != FEATURE_NAMES:
            raise ValueError(
                f"Artifact feature names {artifact_features} do not match expected {FEATURE_NAMES}"
            )

        self._artifact = artifact
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.vectorizer = artifact["vectorizer"]
        logger.info("Phase 5 ML artifact loaded successfully.")

    def predict_claim(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        """
        Runs runtime inference for a single claim-evidence pair:
        1. Ensures artifact is loaded.
        2. Extracts the exact 5 deterministic features matching Phase 5.
        3. Transforms feature vector using persisted TF-IDF vectorizer & scaler.
        4. Predicts class probability using LogisticRegression.predict_proba().
        5. Assigns SUPPORT (1) if P(SUPPORT) >= 0.5, else CONTRADICT (0).
        """
        self.load_artifact()

        # 1. Cosine similarity via BGE embeddings
        cos_sim_vec = extract_bge_cosine_similarities([claim_text], [evidence_text])
        cos_sim = float(cos_sim_vec[0])

        # 2. TF-IDF similarity via persisted vectorizer (transformed, not fitted)
        claim_tfidf = self.vectorizer.transform([claim_text])
        evidence_tfidf = self.vectorizer.transform([evidence_text])
        tfidf_sim = float(sklearn_cosine(claim_tfidf, evidence_tfidf)[0, 0])

        # 3. Keyword overlap ratio
        kw_overlap = compute_keyword_overlap_ratio(claim_text, evidence_text)

        # 4. Claim token length
        c_len = float(compute_token_length(claim_text))

        # 5. Evidence token length
        e_len = float(compute_token_length(evidence_text))

        # Raw feature array matching exact Phase 5 feature order
        X_raw = np.array([[cos_sim, tfidf_sim, kw_overlap, c_len, e_len]], dtype=np.float32)

        # Scale using persisted scaler (transformed, not fitted)
        X_scaled = self.scaler.transform(X_raw)

        # Class probabilities: [P(CONTRADICT), P(SUPPORT)]
        probs = self.model.predict_proba(X_scaled)[0]
        p_support = float(probs[1])

        # Decision rule & confidence calculation
        if p_support >= 0.5:
            verification_status = "SUPPORT"
            confidence_score = p_support
        else:
            verification_status = "CONTRADICT"
            confidence_score = 1.0 - p_support

        return {
            "verification_status": verification_status,
            "confidence_score": round(confidence_score, 4),
            "probability_support": round(p_support, 4),
            "verification_method": "ML-LogisticRegression",
            "feature_values": {
                "cosine_similarity": round(cos_sim, 4),
                "tfidf_similarity": round(tfidf_sim, 4),
                "keyword_overlap_ratio": round(kw_overlap, 4),
                "claim_length": int(c_len),
                "evidence_length": int(e_len)
            }
        }

ml_service = MLInferenceService()
