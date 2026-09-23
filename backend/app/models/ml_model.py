import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base

class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    version = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    feature_names = Column(Text, nullable=False)
    training_samples = Column(Integer, nullable=False)
    test_samples = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    confusion_matrix = Column(Text, nullable=True)
    hyperparameters = Column(Text, nullable=True)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, nullable=False, default=False)
    
    claims = relationship("Claim", back_populates="ml_model")
