import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from ..database import get_db
from ..config import settings
from ..schemas.health import HealthResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    db_status = "unknown"
    ollama_status = "unknown"
    
    # Check Database
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "error"
        
    # Check Ollama gracefully
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                ollama_status = "ok"
            else:
                ollama_status = f"unavailable (HTTP {response.status_code})"
    except Exception as e:
        logger.info(f"Ollama health check: unavailable ({e})")
        ollama_status = "unavailable"
        
    overall_status = "ok" if db_status == "ok" else "error"

    return HealthResponse(
        status=overall_status,
        database=db_status,
        ollama=ollama_status,
        embedding_model=settings.EMBEDDING_MODEL,
    )
