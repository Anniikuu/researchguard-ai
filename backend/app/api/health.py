import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from ..database import get_db
from ..config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    health_status = {
        "status": "ok",
        "services": {
            "database": "unknown",
            "ollama": "unknown"
        }
    }
    
    # Check Database
    try:
        await db.execute(text("SELECT 1"))
        health_status["services"]["database"] = "ok"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["services"]["database"] = "error"
        health_status["status"] = "error"
        
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                health_status["services"]["ollama"] = "ok"
            else:
                health_status["services"]["ollama"] = f"error: {response.status_code}"
                health_status["status"] = "error"
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        health_status["services"]["ollama"] = "error"
        health_status["status"] = "error"
        
    return health_status
