import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ResearchGuard AI API",
    description="API for the ResearchGuard AI RAG system with claim verification.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .utils.exceptions import ResearchGuardException, custom_exception_handler, global_exception_handler
app.add_exception_handler(ResearchGuardException, custom_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting ResearchGuard AI API...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ResearchGuard AI API...")

from .api import health
app.include_router(health.router, prefix="/api")

# Example root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to ResearchGuard AI API"}
