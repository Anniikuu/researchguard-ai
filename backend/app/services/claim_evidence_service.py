import uuid
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..models.answer import Answer
from ..models.claim import Claim, ClaimEvidence
from .claim_extraction_service import extract_claims_from_answer
from .retrieval_service import search_relevant_chunks

logger = logging.getLogger(__name__)

async def process_and_store_claims_for_answer(
    db: AsyncSession,
    answer_id: UUID,
    answer_text: str,
    document_id: Optional[UUID] = None,
    top_k_per_claim: int = 3
) -> List[Dict[str, Any]]:
    """
    Phase 4 Pipeline:
    1. Extracts atomic claims from generated answer.
    2. Runs Phase 2 vector search for EACH claim independently to find claim-specific evidence.
    3. Persists Claim and ClaimEvidence database records with verification_status='UNVERIFIED'.
       (Does NOT make ML verification decisions or train classifiers).
    4. Updates Answer.num_claims.
    """
    # 1. Extract atomic claims
    claim_texts = await extract_claims_from_answer(answer_text)
    if not claim_texts:
        return []

    processed_claims = []

    for idx, claim_text in enumerate(claim_texts):
        # 2. Retrieve claim-specific evidence using Phase 2 pgvector retriever
        retrieved_evidence = await search_relevant_chunks(
            db=db,
            query=claim_text,
            top_k=top_k_per_claim,
            document_id=document_id
        )

        top_similarity = retrieved_evidence[0]["similarity_score"] if retrieved_evidence else 0.0

        # 3. Create Claim record (verification_status='UNVERIFIED', no ML decision yet)
        claim_id = uuid.uuid4()
        claim_obj = Claim(
            id=claim_id,
            answer_id=answer_id,
            claim_index=idx,
            claim_text=claim_text,
            verification_status="UNVERIFIED",
            verification_method="pending_ml",
            cosine_similarity=top_similarity,
            confidence_score=None
        )
        db.add(claim_obj)

        evidence_list = []
        evidence_objects = []

        # 4. Create ClaimEvidence records
        for ev_info in retrieved_evidence:
            chunk_uuid = None
            try:
                chunk_uuid = UUID(ev_info["chunk_id"])
            except (ValueError, KeyError, TypeError):
                chunk_uuid = None

            ev_obj = ClaimEvidence(
                id=uuid.uuid4(),
                claim_id=claim_id,
                chunk_id=chunk_uuid,
                evidence_text=ev_info["content"],
                page_number=ev_info["page_number"],
                similarity_score=ev_info["similarity_score"],
                rank=ev_info.get("rank", 1)
            )
            evidence_objects.append(ev_obj)

            evidence_list.append({
                "chunk_id": ev_info["chunk_id"],
                "document_id": ev_info.get("document_id"),
                "page_number": ev_info["page_number"],
                "evidence_text": ev_info["content"],
                "similarity_score": ev_info["similarity_score"],
                "rank": ev_info.get("rank", 1)
            })

        if evidence_objects:
            db.add_all(evidence_objects)

        processed_claims.append({
            "claim_id": str(claim_id),
            "claim_index": idx,
            "claim_text": claim_text,
            "verification_status": "UNVERIFIED",
            "cosine_similarity": top_similarity,
            "evidence": evidence_list
        })

    # 5. Save to database and update Answer.num_claims
    try:
        stmt = update(Answer).where(Answer.id == answer_id).values(num_claims=len(processed_claims))
        await db.execute(stmt)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to persist Claim and ClaimEvidence records to DB: {e}")
        await db.rollback()

    return processed_claims
