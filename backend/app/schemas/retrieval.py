from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text for vector similarity search")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top chunks to retrieve")
    document_id: Optional[UUID] = Field(default=None, description="Optional document ID filter")

class SearchResultChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    content: str
    token_count: int
    similarity_score: float
    rank: int

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultChunk]
