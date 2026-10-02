import pytest
import httpx
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4
from sqlalchemy import select

from backend.app.config import settings
from backend.app.services.ollama_service import OllamaService, OllamaServiceError
from backend.app.services.rag_service import build_grounded_prompt, execute_grounded_rag
from backend.app.models.question import Question
from backend.app.models.answer import Answer

@pytest.mark.asyncio
async def test_ollama_client_config():
    """Verify Ollama service default settings configuration."""
    service = OllamaService()
    assert service.base_url == settings.OLLAMA_BASE_URL.rstrip("/")
    assert service.model == settings.OLLAMA_MODEL

@pytest.mark.asyncio
async def test_ollama_request_handling_mocked():
    """Test successful Ollama API response parsing using mocked httpx client."""
    mock_json = {
        "model": settings.OLLAMA_MODEL,
        "response": " This is a test grounded response. ",
        "done": True,
        "eval_duration": 500000000,
        "total_duration": 1000000000
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_json
        mock_post.return_value = mock_response

        service = OllamaService()
        result = await service.generate_response(prompt="Hello", system_prompt="Sys")

        assert result["response"] == "This is a test grounded response."
        assert result["eval_duration_ms"] == 500.0
        assert result["total_duration_ms"] == 1000.0


@pytest.mark.asyncio
async def test_ollama_unavailable_handling():
    """Test graceful error handling when Ollama service connection fails."""
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        service = OllamaService()
        with pytest.raises(OllamaServiceError) as exc_info:
            await service.generate_response(prompt="Hello")
        assert exc_info.value.status_code == 503
        assert "Failed to connect to local Ollama server" in exc_info.value.message

@pytest.mark.asyncio
async def test_rag_prompt_construction():
    """Verify prompt formatting includes question, chunks, document IDs, page numbers, and grounding rules."""
    chunks = [
        {
            "chunk_id": str(uuid4()),
            "document_id": "doc-12345",
            "page_number": 3,
            "similarity_score": 0.8912,
            "content": "Deep learning models require quality benchmark datasets."
        },
        {
            "chunk_id": str(uuid4()),
            "document_id": "doc-67890",
            "page_number": 7,
            "similarity_score": 0.7654,
            "content": "pgvector enables vector similarity search within PostgreSQL."
        }
    ]
    question = "What does pgvector enable?"
    prompt = build_grounded_prompt(question, chunks)

    assert "EXTRACTED EVIDENCE CHUNKS:" in prompt
    assert "Document ID: doc-12345 | Page: 3" in prompt
    assert "Deep learning models require quality benchmark datasets." in prompt
    assert "Document ID: doc-67890 | Page: 7" in prompt
    assert "pgvector enables vector similarity search within PostgreSQL." in prompt
    assert "USER QUESTION: What does pgvector enable?" in prompt

@pytest.mark.asyncio
async def test_rag_service_execution_mocked(db_session):
    """Test grounded RAG pipeline execution, prompt passing, source citation, and DB persistence."""
    fake_doc_id = uuid4()
    mock_chunks = [
        {
            "chunk_id": str(uuid4()),
            "document_id": str(fake_doc_id),
            "page_number": 1,
            "chunk_index": 0,
            "content": "ResearchGuard AI uses pgvector for chunk retrieval.",
            "token_count": 8,
            "similarity_score": 0.95,
            "rank": 1
        }
    ]

    mock_ollama_res = {
        "response": "ResearchGuard AI utilizes pgvector for chunk retrieval [Page 1].",
        "model": "qwen2.5:7b-instruct",
        "eval_duration_ms": 250.0,
        "total_duration_ms": 300.0
    }

    with patch("backend.app.services.rag_service.search_relevant_chunks", new_callable=AsyncMock) as mock_search, \
         patch("backend.app.services.rag_service.ollama_service.generate_response", new_callable=AsyncMock) as mock_ollama:
        
        mock_search.return_value = mock_chunks
        mock_ollama.return_value = mock_ollama_res

        result = await execute_grounded_rag(
            db=db_session,
            question="What vector retrieval method is used?",
            top_k=3,
            document_id=fake_doc_id
        )

        assert result["question"] == "What vector retrieval method is used?"
        assert "pgvector for chunk retrieval" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["document_id"] == str(fake_doc_id)
        assert result["sources"][0]["page_number"] == 1
        assert result["sources"][0]["similarity_score"] == 0.95

@pytest.mark.asyncio
async def test_query_api_endpoint_mocked(async_client):
    """Test POST /api/query endpoint with mocked Ollama generation."""
    payload = {
        "question": "What is the primary embedding model?",
        "top_k": 3
    }

    mock_rag_result = {
        "question_id": str(uuid4()),
        "answer_id": str(uuid4()),
        "question": payload["question"],
        "answer": "The primary embedding model is BAAI/bge-small-en-v1.5 [Page 2].",
        "sources": [
            {
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "page_number": 2,
                "chunk_index": 0,
                "content": "Embedding model BAAI/bge-small-en-v1.5 generates 384-dimensional vectors.",
                "similarity_score": 0.92,
                "rank": 1
            }
        ],
        "response_time_ms": 120.5,
        "method": "grounded_rag"
    }

    with patch("backend.app.api.query.execute_grounded_rag", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = mock_rag_result

        response = await async_client.post("/api/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == payload["question"]
        assert "BAAI/bge-small-en-v1.5" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["page_number"] == 2
        assert data["method"] == "grounded_rag"

@pytest.mark.asyncio
async def test_live_ollama_query_execution(db_session):
    """Live integration test calling local Ollama qwen2.5:7b-instruct model."""
    try:
        service = OllamaService()
        res = await service.generate_response(
            prompt="Capital of France?",
            system_prompt="Answer in one word.",
            timeout=10.0
        )
        assert isinstance(res["response"], str)
        assert len(res["response"]) > 0
    except OllamaServiceError:
        pytest.skip("Local Ollama service not running or not responding.")
