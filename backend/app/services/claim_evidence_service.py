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
from ..ml.inference import ml_service

logger = logging.getLogger(__name__)

async def process_and_store_claims_for_answer(
    db: AsyncSession,
    answer_id: UUID,
    answer_text: str,
    document_id: Optional[UUID] = None,
    top_k_per_claim: int = 3
) -> List[Dict[str, Any]]:
    """
    Phase 6 Pipeline:
    1. Extracts atomic claims from generated answer.
    2. Runs Phase 2 vector search for EACH claim independently to find claim-specific evidence.
    3. If usable evidence exists:
       - Selects primary top-ranked evidence chunk.
       - Runs Phase 5 trained Logistic Regression via MLInferenceService.
       - Assigns SUPPORT or CONTRADICT verification_status and model confidence score.
       - Stores feature values (cosine_similarity, tfidf_similarity, keyword_overlap, claim_length, evidence_length).
    4. If NO usable evidence exists:
       - Preserves safe insufficient-evidence behavior: status='UNVERIFIED', method='insufficient_evidence'.
       - Does NOT invoke ML model or assign CONTRADICT.
    5. Persists Claim and ClaimEvidence database records and updates Answer.num_claims.
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
        primary_evidence_text = (
            retrieved_evidence[0]["content"].strip()
            if retrieved_evidence and retrieved_evidence[0].get("content")
            else ""
        )

        # 3. Determine verification status via ML model if usable evidence exists
        if primary_evidence_text:
            try:
                ml_res = ml_service.predict_claim(claim_text, primary_evidence_text)
                verification_status = ml_res["verification_status"]
                confidence_score = ml_res["confidence_score"]
                verification_method = ml_res["verification_method"]
                feats = ml_res["feature_values"]
                cosine_sim = feats["cosine_similarity"]
                tfidf_sim = feats["tfidf_similarity"]
                kw_overlap = feats["keyword_overlap_ratio"]
                claim_len = feats["claim_length"]
                evidence_len = feats["evidence_length"]
            except Exception as e:
                logger.error(f"ML verification failed for claim {idx}: {e}")
                verification_status = "UNVERIFIED"
                confidence_score = None
                verification_method = "ml_error"
                cosine_sim = top_similarity
                tfidf_sim = None
                kw_overlap = None
                claim_len = None
                evidence_len = None
        else:
            # Insufficient evidence: do NOT run ML inference, do NOT assign CONTRADICT
            verification_status = "UNVERIFIED"
            confidence_score = None
            verification_method = "insufficient_evidence"
            cosine_sim = top_similarity
            tfidf_sim = None
            kw_overlap = None
            claim_len = None
            evidence_len = None

        # 4. Create Claim record
        claim_id = uuid.uuid4()
        claim_obj = Claim(
            id=claim_id,
            answer_id=answer_id,
            claim_index=idx,
            claim_text=claim_text,
            verification_status=verification_status,
            confidence_score=confidence_score,
            verification_method=verification_method,
            cosine_similarity=cosine_sim,
            tfidf_similarity=tfidf_sim,
            keyword_overlap=kw_overlap,
            claim_length=claim_len,
            evidence_length=evidence_len
        )
        db.add(claim_obj)

        evidence_list = []
        evidence_objects = []

        # 5. Create ClaimEvidence records
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
            "verification_status": verification_status,
            "confidence_score": confidence_score,
            "verification_method": verification_method,
            "cosine_similarity": cosine_sim,
            "tfidf_similarity": tfidf_sim,
            "keyword_overlap": kw_overlap,
            "claim_length": claim_len,
            "evidence_length": evidence_len,
            "evidence": evidence_list
        })

    # 6. Save to database and update Answer.num_claims
    try:
        stmt = update(Answer).where(Answer.id == answer_id).values(num_claims=len(processed_claims))
        await db.execute(stmt)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to persist Claim and ClaimEvidence records to DB: {e}")
        await db.rollback()

    return processed_claims

