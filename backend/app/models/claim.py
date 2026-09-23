import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base

class Claim(Base):
    __tablename__ = "claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    answer_id = Column(UUID(as_uuid=True), ForeignKey("answers.id", ondelete="CASCADE"), nullable=False)
    claim_index = Column(Integer, nullable=False)
    claim_text = Column(Text, nullable=False)
    verification_status = Column(String, nullable=False) # SUPPORTED, UNSUPPORTED, UNVERIFIABLE
    confidence_score = Column(Float, nullable=True)
    verification_method = Column(String, nullable=False) # "baseline_threshold", "logistic_regression"
    ml_model_id = Column(UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="SET NULL"), nullable=True)
    
    cosine_similarity = Column(Float, nullable=True)
    tfidf_similarity = Column(Float, nullable=True)
    keyword_overlap = Column(Float, nullable=True)
    claim_length = Column(Integer, nullable=True)
    evidence_length = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    answer = relationship("Answer", back_populates="claims")
    evidence = relationship("ClaimEvidence", back_populates="claim", cascade="all, delete-orphan")
    ml_model = relationship("MLModel", back_populates="claims")

class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True)
    evidence_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    similarity_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    claim = relationship("Claim", back_populates="evidence")
    chunk = relationship("DocumentChunk", back_populates="evidence")
