from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class ResearchGuardException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

class DocumentNotFoundError(ResearchGuardException):
    def __init__(self, message: str = "Document not found"):
        super().__init__(message, status_code=404)

async def custom_exception_handler(request: Request, exc: ResearchGuardException):
    logger.error(f"Custom Exception: {exc.message} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )

async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled Exception - Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
