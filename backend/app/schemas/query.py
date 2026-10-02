from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer using grounded RAG")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of evidence chunks to retrieve")
    document_id: Optional[UUID] = Field(default=None, description="Optional specific document ID filter")

class SourceCitation(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    content: str
    similarity_score: float
    rank: int = 1

class ClaimEvidenceSchema(BaseModel):
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    page_number: int
    evidence_text: str
    similarity_score: float
    rank: int = 1

class ClaimSchema(BaseModel):
    claim_id: str
    claim_index: int
    claim_text: str
    verification_status: str = "UNVERIFIED"
    cosine_similarity: Optional[float] = None
    evidence: List[ClaimEvidenceSchema] = Field(default_factory=list)

class QueryResponse(BaseModel):
    question_id: str
    answer_id: str
    question: str
    answer: str
    sources: List[SourceCitation]
    claims: List[ClaimSchema] = Field(default_factory=list)
    response_time_ms: float
    method: str = "grounded_rag"

