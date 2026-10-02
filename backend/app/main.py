import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .utils.exceptions import ResearchGuardException, custom_exception_handler, global_exception_handler
from .api import health, documents, retrieval, query

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ResearchGuard AI API...")
    yield
    logger.info("Shutting down ResearchGuard AI API...")

app = FastAPI(
    title="ResearchGuard AI API",
    description="API for the ResearchGuard AI RAG system with claim verification.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(ResearchGuardException, custom_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(retrieval.router, prefix="/api")
app.include_router(query.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to ResearchGuard AI API"}
