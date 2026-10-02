import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.document import Document
from ..models.question import Question
from ..models.answer import Answer
from .retrieval_service import search_relevant_chunks
from .ollama_service import ollama_service

from .claim_evidence_service import process_and_store_claims_for_answer

logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT = (
    "You are ResearchGuard AI, an authoritative, precise academic document assistant.\n"
    "Your task is to answer the user's question using ONLY the provided document evidence chunks below.\n\n"
    "STRICT GROUNDING RULES:\n"
    "1. Answer using ONLY the supplied evidence. Do NOT use unrestricted outside knowledge.\n"
    "2. Do NOT invent facts, speculate, or extrapolate beyond the provided text.\n"
    "3. If the provided evidence does not contain sufficient information to answer the question, explicitly state: "
    "\"The provided documents do not contain enough information to answer this question.\"\n"
    "4. Do NOT treat the user's question as evidence.\n"
    "5. Do NOT fabricate citations.\n"
    "6. Cite the relevant source and page numbers (e.g. [Page X]) supplied in the evidence when making statements."
)

def build_grounded_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs a deterministic RAG prompt embedding the user question and top-k evidence chunks.
    """
    if not chunks:
        context_str = "NO RELEVANT DOCUMENT EVIDENCE FOUND IN DATABASE."
    else:
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            doc_info = f"Document ID: {chunk.get('document_id', 'Unknown')}"
            page_info = f"Page: {chunk.get('page_number', 'Unknown')}"
            score_info = f"Similarity: {chunk.get('similarity_score', 0.0):.4f}"
            content = chunk.get("content", "").strip()
            formatted_chunks.append(
                f"--- EVIDENCE CHUNK #{i} [{doc_info} | {page_info} | {score_info}] ---\n{content}"
            )
        context_str = "\n\n".join(formatted_chunks)

    prompt = (
        f"EXTRACTED EVIDENCE CHUNKS:\n\n"
        f"{context_str}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"GROUNDED ANSWER:"
    )
    return prompt

async def execute_grounded_rag(
    db: AsyncSession,
    question: str,
    top_k: int = 5,
    document_id: Optional[UUID] = None
) -> Dict[str, Any]:
    """
    Executes Phase 3 Grounded RAG flow + Phase 4 Claim Extraction & Evidence Pipeline:
    1. Vector retrieval using Phase 2 search_relevant_chunks.
    2. Grounded prompt construction.
    3. LLM generation via Ollama client (qwen2.5:7b-instruct).
    4. Source/page citation extraction.
    5. Persistence of Question and Answer models.
    6. Phase 4: Atomic Claim Extraction & Claim-Specific Evidence Retrieval + Persistence.
    """
    start_time = time.perf_counter()

    # 1. Retrieve evidence chunks from vector store
    retrieved_chunks = await search_relevant_chunks(
        db=db,
        query=question,
        top_k=top_k,
        document_id=document_id
    )

    # 2. Build grounded prompt
    prompt = build_grounded_prompt(question, retrieved_chunks)
    context_used = prompt

    # 3. Call Ollama for answer generation
    ollama_res = await ollama_service.generate_response(
        prompt=prompt,
        system_prompt=GROUNDED_SYSTEM_PROMPT
    )

    answer_text = ollama_res.get("response", "")
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Format source citations
    sources = []
    for chunk in retrieved_chunks:
        sources.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "similarity_score": chunk["similarity_score"],
            "rank": chunk.get("rank", 1)
        })

    # 5. Persist Question & Answer records if target document exists
    target_doc_id: Optional[UUID] = document_id
    if target_doc_id is None and retrieved_chunks:
        try:
            target_doc_id = UUID(retrieved_chunks[0]["document_id"])
        except (ValueError, KeyError, TypeError):
            target_doc_id = None

    if target_doc_id is None:
        # Check if any document exists in the DB to associate the question
        stmt = select(Document.id).limit(1)
        res = await db.execute(stmt)
        target_doc_id = res.scalar_one_or_none()

    question_id = uuid.uuid4()
    answer_id = uuid.uuid4()

    if target_doc_id is not None:
        try:
            q_record = Question(
                id=question_id,
                document_id=target_doc_id,
                question_text=question,
                method="grounded_rag"
            )
            db.add(q_record)

            a_record = Answer(
                id=answer_id,
                question_id=question_id,
                answer_text=answer_text,
                context_used=context_used,
                num_claims=0,
                num_supported=0,
                num_unsupported=0,
                response_time_ms=elapsed_ms,
                method="grounded_rag"
            )
            db.add(a_record)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist Question/Answer record to DB: {e}")
            await db.rollback()

    # 6. Phase 4: Claim extraction + Claim-specific evidence retrieval + persistence
    claims = []
    try:
        claims = await process_and_store_claims_for_answer(
            db=db,
            answer_id=answer_id,
            answer_text=answer_text,
            document_id=target_doc_id,
            top_k_per_claim=3
        )
    except Exception as e:
        logger.error(f"Error during Phase 4 claim evidence processing: {e}")

    return {
        "question_id": str(question_id),
        "answer_id": str(answer_id),
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "claims": claims,
        "response_time_ms": round(elapsed_ms, 2),
        "method": "grounded_rag"
    }

