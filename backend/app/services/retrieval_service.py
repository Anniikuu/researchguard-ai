import math
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..models.document import Document
from ..models.chunk import DocumentChunk
from .pdf_service import extract_text_from_pdf_bytes, PDFExtractionError
from .chunking_service import chunk_pages_data
from .embedding_service import embedding_service

logger = logging.getLogger(__name__)

async def process_and_store_document(
    db: AsyncSession,
    filename: str,
    original_filename: str,
    pdf_bytes: bytes
) -> Document:
    """
    Ingests a PDF: extracts page text, chunks text, generates embeddings, and saves to database.
    Processes CPU operations first before executing database transaction.
    """
    file_size = len(pdf_bytes)
    
    # 1. Extract text page-by-page (CPU bound)
    pages_data = extract_text_from_pdf_bytes(pdf_bytes)
    page_count = len(pages_data)
    
    # 2. Chunk pages (CPU bound)
    chunks_data = chunk_pages_data(pages_data)
    
    # 3. Generate embeddings in batch (CPU/Model bound)
    texts_to_embed = [c["content"] for c in chunks_data]
    embeddings = embedding_service.generate_embeddings_batch(texts_to_embed)
    
    # 4. Construct Document record
    doc_id = uuid.uuid4()
    doc_record = Document(
        id=doc_id,
        filename=filename,
        original_filename=original_filename,
        file_size_bytes=file_size,
        page_count=page_count,
        chunk_count=len(chunks_data),
        status="completed",
        uploaded_at=datetime.now(),
        processed_at=datetime.now()
    )
    
    # 5. Construct DocumentChunk records
    chunk_objects = []
    for idx, c_info in enumerate(chunks_data):
        emb_vec = embeddings[idx] if idx < len(embeddings) else [0.0] * 384
        chunk_obj = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=c_info["chunk_index"],
            page_number=c_info["page_number"],
            content=c_info["content"],
            embedding=emb_vec,
            token_count=c_info["token_count"],
            created_at=datetime.now()
        )
        chunk_objects.append(chunk_obj)
        
    # 6. Save to database in a single atomic transaction
    try:
        db.add(doc_record)
        if chunk_objects:
            db.add_all(chunk_objects)
        await db.commit()
        await db.refresh(doc_record)
        return doc_record
    except Exception as e:
        import traceback
        logger.error(f"Failed to commit document {original_filename} to DB: {e}\n{traceback.format_exc()}")
        await db.rollback()
        raise e

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Helper for cosine similarity calculation."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

async def search_relevant_chunks(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
    document_id: Optional[UUID] = None
) -> List[Dict[str, Any]]:
    """
    Performs vector similarity search against stored DocumentChunk embeddings.
    """
    if not query.strip():
        return []
        
    query_vector = embedding_service.generate_embedding(query)
    
    # Query database chunks
    stmt = select(DocumentChunk)
    if document_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == document_id)
        
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    
    if not chunks:
        return []
        
    # Calculate similarity score for each chunk
    scored_chunks = []
    for chunk in chunks:
        emb = chunk.embedding
        if isinstance(emb, list):
            score = _cosine_similarity(query_vector, emb)
        else:
            try:
                score = _cosine_similarity(query_vector, list(emb))
            except Exception:
                score = 0.0
                
        scored_chunks.append({
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "similarity_score": round(float(score), 4),
        })
        
    # Rank by similarity_score descending
    scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Assign ranks and slice top_k
    top_results = scored_chunks[:top_k]
    for idx, item in enumerate(top_results):
        item["rank"] = idx + 1
        
    return top_results
