import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base

class EvaluationQuestion(Base):
    __tablename__ = "evaluation_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    question_text = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=False)
    expected_claims = Column(Text, nullable=False)
    difficulty = Column(String, nullable=True)
    category = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="evaluation_questions")
