import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base

class Answer(Base):
    __tablename__ = "answers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id = Column(UUID(as_uuid=True), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, unique=True)
    answer_text = Column(Text, nullable=False)
    context_used = Column(Text, nullable=False)
    num_claims = Column(Integer, nullable=False, default=0)
    num_supported = Column(Integer, nullable=False, default=0)
    num_unsupported = Column(Integer, nullable=False, default=0)
    avg_confidence = Column(Float, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    method = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    question = relationship("Question", back_populates="answer")
    claims = relationship("Claim", back_populates="answer", cascade="all, delete-orphan")
