from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    database: str
    ollama: str
    embedding_model: str
