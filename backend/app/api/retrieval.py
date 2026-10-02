from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..schemas.retrieval import SearchRequest, SearchResponse, SearchResultChunk
from ..services.retrieval_service import search_relevant_chunks

router = APIRouter()

@router.post("/retrieval/search", response_model=SearchResponse)
async def search_retrieval(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        raw_results = await search_relevant_chunks(
            db=db,
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id
        )
        
        results = [SearchResultChunk(**item) for item in raw_results]
        
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {str(e)}"
        )
