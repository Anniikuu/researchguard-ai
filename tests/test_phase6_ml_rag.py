import pytest
import os
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from backend.app.ml.inference import ml_service, MLInferenceService, FEATURE_NAMES
from backend.app.schemas.query import ClaimSchema, QueryResponse

def test_ml_artifact_loading():
    """Verify Phase 5 artifact exists, loads properly, and contains required keys & feature order."""
    service = MLInferenceService()
    service.load_artifact()
    
    assert service.model is not None
    assert service.scaler is not None
    assert service.vectorizer is not None
    assert service._artifact["feature_names"] == FEATURE_NAMES
    assert service._artifact["feature_names"] == [
        "cosine_similarity",
        "tfidf_similarity",
        "keyword_overlap_ratio",
        "claim_length",
        "evidence_length"
    ]

def test_runtime_preprocessing_transform_not_fitted():
    """Verify vectorizer and scaler are invoked with transform(), NOT fit()."""
    service = MLInferenceService()
    service.load_artifact()
    
    with patch.object(service.vectorizer, 'transform', wraps=service.vectorizer.transform) as mock_vec_tf, \
         patch.object(service.scaler, 'transform', wraps=service.scaler.transform) as mock_scaler_tf:
        
        res = service.predict_claim(
            claim_text="Cell division is regulated by cyclins.",
            evidence_text="Cyclins are a family of proteins that control the progression of cells through the cell cycle."
        )
        
        assert mock_vec_tf.called
        assert mock_scaler_tf.called
        assert res["verification_status"] in ["SUPPORT", "CONTRADICT"]
        assert 0.0 <= res["confidence_score"] <= 1.0

def test_ml_inference_decision_and_confidence():
    """Verify predict_claim returns probability, valid confidence, and correct label mapping."""
    claim = "RNA polymerase synthesizes mRNA from a DNA template."
    evidence = "Transcription is the process of copying a segment of DNA into RNA by RNA polymerase."
    
    res = ml_service.predict_claim(claim, evidence)
    
    assert "verification_status" in res
    assert "confidence_score" in res
    assert "probability_support" in res
    assert res["verification_method"] == "ML-LogisticRegression"
    
    status = res["verification_status"]
    conf = res["confidence_score"]
    p_sup = res["probability_support"]
    
    if p_sup >= 0.5:
        assert status == "SUPPORT"
        assert abs(conf - p_sup) < 1e-3
    else:
        assert status == "CONTRADICT"
        assert abs(conf - (1.0 - p_sup)) < 1e-3

    feats = res["feature_values"]
    assert "cosine_similarity" in feats
    assert "tfidf_similarity" in feats
    assert "keyword_overlap_ratio" in feats
    assert "claim_length" in feats
    assert "evidence_length" in feats

@pytest.mark.asyncio
async def test_insufficient_evidence_behavior():
    """Verify ML model is NOT invoked and status is UNVERIFIED (not CONTRADICT) when evidence is missing."""
    from backend.app.services.claim_evidence_service import process_and_store_claims_for_answer
    
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    
    with patch("backend.app.services.claim_evidence_service.extract_claims_from_answer", return_value=["Some extracted claim."]), \
         patch("backend.app.services.claim_evidence_service.search_relevant_chunks", return_value=[]), \
         patch("backend.app.services.claim_evidence_service.ml_service.predict_claim") as mock_predict:
        
        claims = await process_and_store_claims_for_answer(
            db=mock_db,
            answer_id="00000000-0000-0000-0000-000000000001",
            answer_text="Some extracted claim.",
            document_id=None
        )
        
        # ML model must NOT be called when evidence is empty/missing
        assert not mock_predict.called
        assert len(claims) == 1
        assert claims[0]["verification_status"] == "UNVERIFIED"
        assert claims[0]["verification_status"] != "CONTRADICT"
        assert claims[0]["verification_method"] == "insufficient_evidence"
        assert claims[0]["confidence_score"] is None

@pytest.mark.asyncio
async def test_rag_ml_integration_flow():
    """Verify end-to-end claim evidence processing invokes ML service when evidence exists."""
    from backend.app.services.claim_evidence_service import process_and_store_claims_for_answer
    
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_chunks = [{
        "chunk_id": "00000000-0000-0000-0000-000000000002",
        "document_id": "00000000-0000-0000-0000-000000000003",
        "page_number": 1,
        "content": "DNA contains genetic instructions used in development.",
        "similarity_score": 0.85,
        "rank": 1
    }]
    
    with patch("backend.app.services.claim_evidence_service.extract_claims_from_answer", return_value=["DNA contains genetic code."]), \
         patch("backend.app.services.claim_evidence_service.search_relevant_chunks", return_value=mock_chunks):
        
        claims = await process_and_store_claims_for_answer(
            db=mock_db,
            answer_id="00000000-0000-0000-0000-000000000001",
            answer_text="DNA contains genetic code.",
            document_id=None
        )
        
        assert len(claims) == 1
        c = claims[0]
        assert c["verification_status"] in ["SUPPORT", "CONTRADICT"]
        assert c["verification_method"] == "ML-LogisticRegression"
        assert c["confidence_score"] is not None
        assert c["cosine_similarity"] is not None
        assert c["tfidf_similarity"] is not None
        assert c["keyword_overlap"] is not None
        assert len(c["evidence"]) == 1
        assert c["evidence"][0]["evidence_text"] == "DNA contains genetic instructions used in development."

@pytest.mark.asyncio
async def test_api_scifact_experiment_endpoint():
    """Verify GET /api/experiments/scifact returns Phase 5 experiment metrics JSON."""
    from backend.app.api.experiments import get_scifact_experiment_results
    
    res = await get_scifact_experiment_results()
    
    assert "experiment" in res
    assert "aggregate_results" in res
    assert "baseline_cosine_threshold" in res["aggregate_results"]
    assert "proposed_logistic_regression" in res["aggregate_results"]
    
    baseline = res["aggregate_results"]["baseline_cosine_threshold"]
    lr = res["aggregate_results"]["proposed_logistic_regression"]
    
    assert baseline["accuracy"]["mean"] == 0.6456
    assert lr["precision"]["mean"] == 0.6892
