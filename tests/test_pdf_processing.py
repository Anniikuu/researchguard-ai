import io
import fitz  # PyMuPDF
import pytest
from httpx import AsyncClient

from backend.app.services.pdf_service import extract_text_from_pdf_bytes, PDFExtractionError
from backend.app.services.chunking_service import chunk_pages_data
from backend.app.services.embedding_service import embedding_service

def create_sample_pdf_bytes() -> bytes:
    """Helper to generate a clean 2-page PDF in memory."""
    doc = fitz.open()
    
    page1 = doc.new_page()
    page1.insert_text((50, 50), "ResearchGuard AI is a local Retrieval-Augmented Generation system. Page 1 contains core concepts.")
    
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Logistic Regression is employed as the primary machine learning classifier for verifying scientific claims.")
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def test_pdf_extraction():
    """Verify PyMuPDF extracts text page-by-page and preserves page numbers."""
    pdf_bytes = create_sample_pdf_bytes()
    pages = extract_text_from_pdf_bytes(pdf_bytes)
    
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "ResearchGuard AI" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Logistic Regression" in pages[1]["text"]

def test_chunking_service():
    """Verify deterministic page-aware chunking preserves page numbers and indices."""
    pages = [
        {"page_number": 1, "text": "First page text content for testing chunking."},
        {"page_number": 2, "text": "Second page text content for verifying chunking preservation."}
    ]
    
    chunks = chunk_pages_data(pages, chunk_size=30, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all("page_number" in c for c in chunks)
    assert all("chunk_index" in c for c in chunks)
    assert chunks[0]["page_number"] == 1

def test_embedding_service_dimension():
    """Verify BAAI/bge-small-en-v1.5 embedding output dimension is 384."""
    vec = embedding_service.generate_embedding("Test scientific query")
    assert isinstance(vec, list)
    assert len(vec) == 384

@pytest.mark.asyncio
async def test_pdf_ingestion_and_vector_search_api(async_client: AsyncClient):
    """Integration test: Upload PDF -> Ingest -> Perform vector search retrieval."""
    pdf_bytes = create_sample_pdf_bytes()
    
    # 1. Upload PDF
    files = {"file": ("test_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_resp = await async_client.post("/api/documents/upload", files=files)
    
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    assert doc_data["status"] == "completed"
    assert doc_data["page_count"] == 2
    assert doc_data["chunk_count"] >= 2
    doc_id = doc_data["id"]
    
    # 2. Perform Retrieval Search
    search_payload = {
        "query": "What machine learning classifier is used?",
        "top_k": 2,
        "document_id": doc_id
    }
    search_resp = await async_client.post("/api/retrieval/search", json=search_payload)
    
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["query"] == search_payload["query"]
    assert search_data["total_results"] > 0
    
    first_result = search_data["results"][0]
    assert first_result["rank"] == 1
    assert first_result["document_id"] == doc_id
    assert "Logistic Regression" in first_result["content"]
    assert first_result["page_number"] == 2
    assert isinstance(first_result["similarity_score"], float)

@pytest.mark.asyncio
async def test_upload_invalid_file_type(async_client: AsyncClient):
    """Verify invalid non-PDF upload fails with 400."""
    files = {"file": ("test.txt", io.BytesIO(b"Hello text"), "text/plain")}
    resp = await async_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 400
    assert "Only PDF files are supported" in resp.json()["detail"]
