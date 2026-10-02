import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..schemas.query import QueryRequest, QueryResponse
from ..services.rag_service import execute_grounded_rag
from ..services.ollama_service import OllamaServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db)
) -> QueryResponse:
    """
    POST /api/query
    
    Executes a Grounded RAG query over indexed document chunks:
    1. Vector similarity search retrieves relevant evidence chunks.
    2. Constructs a grounded system prompt.
    3. Calls local Ollama (qwen2.5:7b-instruct) to generate a grounded answer.
    4. Returns answer, supporting sources (with document ID and page numbers), and metrics.
    """
    try:
        result = await execute_grounded_rag(
            db=db,
            question=payload.question,
            top_k=payload.top_k,
            document_id=payload.document_id
        )
        return QueryResponse(**result)
    except OllamaServiceError as e:
        logger.error(f"Ollama Service error during query execution: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Unexpected error in /api/query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )
