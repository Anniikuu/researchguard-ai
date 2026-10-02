import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from ..database import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=True)  # 384 dimensions for BAAI/bge-small-en-v1.5
    token_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.now)

    document = relationship("Document", back_populates="chunks")
    evidence = relationship("ClaimEvidence", back_populates="chunk", cascade="all, delete-orphan")
