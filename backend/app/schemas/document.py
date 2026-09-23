from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import Optional

class DocumentBase(BaseModel):
    filename: str
    original_filename: str

class DocumentCreate(DocumentBase):
    file_size_bytes: int

class DocumentResponse(DocumentBase):
    id: UUID
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    uploaded_at: datetime
    processed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
