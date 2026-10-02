import pytest
import httpx
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4
from sqlalchemy import select

from backend.app.services.claim_extraction_service import (
    extract_claims_from_answer,
    _clean_and_parse_json,
    _fallback_sentence_splitter
)
from backend.app.services.claim_evidence_service import process_and_store_claims_for_answer
from backend.app.models.claim import Claim, ClaimEvidence

@pytest.mark.asyncio
async def test_claim_extraction_from_known_answer_mocked():
    """Verify JSON parsing and atomic claim separation from structured LLM response."""
    mock_json_response = {
        "response": '{\n  "claims": [\n    "Logistic regression is a supervised machine learning algorithm.",\n    "It is commonly used for binary classification."\n  ]\n}'
    }

    with patch("backend.app.services.claim_extraction_service.ollama_service.generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_json_response

        answer = "Logistic regression is a supervised machine learning algorithm. It is commonly used for binary classification."
        claims = await extract_claims_from_answer(answer)

        assert len(claims) == 2
        assert claims[0] == "Logistic regression is a supervised machine learning algorithm."
        assert claims[1] == "It is commonly used for binary classification."
        # Verify no verification decision label was generated
        for c in claims:
            assert "SUPPORTED" not in c
            assert "UNSUPPORTED" not in c

@pytest.mark.asyncio
async def test_no_verification_labels_produced():
    """Ensure claim extraction isolates claims without making truth/verification decisions."""
    json_text = '{"claims": ["pgvector adds vector similarity search to PostgreSQL."]}'
    parsed = _clean_and_parse_json(json_text)
    assert parsed == ["pgvector adds vector similarity search to PostgreSQL."]
    assert "SUPPORTED" not in parsed[0]
    assert "UNSUPPORTED" not in parsed[0]

@pytest.mark.asyncio
async def test_fallback_sentence_splitter():
    """Test deterministic sentence-based fallback when LLM output is malformed or not JSON."""
    raw_text = "PyMuPDF extracts page text from PDFs. BGE embeddings generate 384d vectors. [Page 2]"
    fallback_claims = _fallback_sentence_splitter(raw_text)

    assert len(fallback_claims) == 2
    assert "PyMuPDF extracts page text from PDFs." in fallback_claims
    assert "BGE embeddings generate 384d vectors." in fallback_claims

@pytest.mark.asyncio
async def test_empty_and_insufficient_answer_handling():
    """Verify empty text or 'insufficient info' statements yield zero claims."""
    assert await extract_claims_from_answer("") == []
    assert await extract_claims_from_answer("The provided documents do not contain enough information to answer this question.") == []

@pytest.mark.asyncio
async def test_claim_evidence_pipeline_and_db_persistence(db_session):
    """Test full Phase 4 claim extraction, claim-specific evidence search, and DB persistence."""
    mock_claims = [
        "BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings.",
        "pgvector handles cosine similarity retrieval."
    ]

    fake_answer_id = uuid4()
    fake_doc_id = uuid4()
    fake_chunk_id = uuid4()

    mock_evidence = [
        {
            "chunk_id": str(fake_chunk_id),
            "document_id": str(fake_doc_id),
            "page_number": 4,
            "chunk_index": 1,
            "content": "Embedding model BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings.",
            "similarity_score": 0.9412,
            "rank": 1
        }
    ]

    with patch("backend.app.services.claim_evidence_service.extract_claims_from_answer", new_callable=AsyncMock) as mock_extract, \
         patch("backend.app.services.claim_evidence_service.search_relevant_chunks", new_callable=AsyncMock) as mock_search:

        mock_extract.return_value = mock_claims
        mock_search.return_value = mock_evidence

        results = await process_and_store_claims_for_answer(
            db=db_session,
            answer_id=fake_answer_id,
            answer_text="Dummy answer text",
            document_id=fake_doc_id,
            top_k_per_claim=2
        )

        assert len(results) == 2
        assert results[0]["claim_text"] == mock_claims[0]
        assert results[0]["verification_status"] == "UNVERIFIED"
        assert len(results[0]["evidence"]) == 1
        assert results[0]["evidence"][0]["page_number"] == 4
        assert results[0]["evidence"][0]["similarity_score"] == 0.9412

@pytest.mark.asyncio
async def test_query_api_includes_claims(async_client):
    """Verify POST /api/query endpoint response schema includes extracted claims and per-claim evidence."""
    payload = {
        "question": "What does pgvector provide?",
        "top_k": 3
    }

    mock_rag_result = {
        "question_id": str(uuid4()),
        "answer_id": str(uuid4()),
        "question": payload["question"],
        "answer": "pgvector provides vector similarity search in PostgreSQL [Page 1].",
        "sources": [],
        "claims": [
            {
                "claim_id": str(uuid4()),
                "claim_index": 0,
                "claim_text": "pgvector provides vector similarity search in PostgreSQL.",
                "verification_status": "UNVERIFIED",
                "cosine_similarity": 0.91,
                "evidence": [
                    {
                        "chunk_id": str(uuid4()),
                        "document_id": str(uuid4()),
                        "page_number": 1,
                        "evidence_text": "pgvector extension adds vector similarity search capabilities.",
                        "similarity_score": 0.91,
                        "rank": 1
                    }
                ]
            }
        ],
        "response_time_ms": 150.0,
        "method": "grounded_rag"
    }

    with patch("backend.app.api.query.execute_grounded_rag", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = mock_rag_result

        res = await async_client.post("/api/query", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "claims" in data
        assert len(data["claims"]) == 1
        assert data["claims"][0]["claim_text"] == "pgvector provides vector similarity search in PostgreSQL."
        assert data["claims"][0]["verification_status"] == "UNVERIFIED"
        assert data["claims"][0]["evidence"][0]["page_number"] == 1
