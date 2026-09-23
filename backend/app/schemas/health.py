from pydantic import BaseModel
from typing import Optional

class HealthResponse(BaseModel):
    status: str
    database: str
    ollama: str
    embedding_model: str
