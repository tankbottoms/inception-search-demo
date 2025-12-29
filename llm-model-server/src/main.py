"""
Python GPU Inference Service - FastAPI Application
Mirrors the TypeScript ONNX service API
"""
import time
import logging
import subprocess
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try to import onnxruntime for provider info
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from config import settings
from embedding_service_torch import embedding_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# Request/Response Models
# ============================================================

class QueryRequest(BaseModel):
    text: str


class QueryResponse(BaseModel):
    embedding: List[float]


class TextRequest(BaseModel):
    id: int
    text: str


class ChunkEmbedding(BaseModel):
    chunk_number: int
    chunk: str
    embedding: List[float]


class TextResponse(BaseModel):
    id: int
    embeddings: List[ChunkEmbedding]


class BatchTextRequest(BaseModel):
    documents: List[TextRequest]


class TimingResponse(BaseModel):
    total_ms: float
    chunking_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    postprocess_ms: Optional[float] = None


class BatchResponse(BaseModel):
    results: List[TextResponse]
    timing: TimingResponse


# ============================================================
# Hardware Detection
# ============================================================

def get_gpu_info():
    """Get GPU information via nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                return {
                    "device_name": parts[0].strip(),
                    "memory_total": int(parts[1].strip()),
                    "memory_free": int(parts[2].strip()),
                    "driver_version": parts[3].strip(),
                }
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
    return None


# ============================================================
# Lifespan Management
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize service on startup"""
    logger.info(f"Starting Python GPU Inference Service on port {settings.port}")
    if ORT_AVAILABLE:
        logger.info(f"ONNX Runtime version: {ort.__version__}")
        logger.info(f"Available providers: {ort.get_available_providers()}")

    gpu_info = get_gpu_info()
    if gpu_info:
        logger.info(f"GPU: {gpu_info['device_name']}")
        logger.info(f"GPU Memory: {gpu_info['memory_free']}/{gpu_info['memory_total']} MB free")

    try:
        await embedding_service.initialize()
        logger.info("Embedding service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        logger.warning("Service will attempt to initialize on first request")

    yield

    logger.info("Shutting down service")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Inception GPU Inference Service",
    description="GPU-accelerated ONNX inference for embeddings",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health & Status Endpoints
# ============================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    gpu_info = get_gpu_info()
    status = embedding_service.get_status()

    return {
        "status": "ok",
        "version": "2.0.0-gpu",
        "provider": status["provider"],
        "device": gpu_info["device_name"] if gpu_info else "CPU",
        "model": status["model_id"] or "not loaded",
        "initialized": status["initialized"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }


@app.get("/status")
async def status():
    """Detailed status endpoint"""
    gpu_info = get_gpu_info()
    service_status = embedding_service.get_status()

    return {
        "service": {
            "version": "2.0.0-gpu",
            "initialized": service_status["initialized"],
            "modelId": service_status["model_id"],
            "embeddingDim": service_status["embedding_dim"],
        },
        "hardware": {
            "provider": service_status["provider"],
            "backend": service_status.get("backend", "unknown"),
            "device": gpu_info["device_name"] if gpu_info else "CPU",
            "memoryTotal": gpu_info["memory_total"] if gpu_info else None,
            "memoryFree": gpu_info["memory_free"] if gpu_info else None,
            "onnxrtVersion": ort.__version__ if ORT_AVAILABLE else None,
            "availableProviders": ort.get_available_providers() if ORT_AVAILABLE else [],
            "cudaAvailable": service_status.get("cuda_available", False),
        },
        "config": {
            "maxTokens": settings.max_tokens,
            "maxBatchSize": settings.max_batch_size,
            "processingBatchSize": settings.processing_batch_size,
        },
    }


# ============================================================
# Embedding Endpoints
# ============================================================

@app.post("/api/v1/embed/query", response_model=QueryResponse)
async def embed_query(request: QueryRequest):
    """Generate embedding for a search query"""
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        embedding, timing = await embedding_service.generate_query_embedding(request.text)
        return QueryResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"Query embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embed/text", response_model=TextResponse)
async def embed_text(request: TextRequest):
    """Generate embeddings for a document with chunking"""
    if request.id is None:
        raise HTTPException(status_code=400, detail="id is required")
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        chunks, timing = await embedding_service.generate_text_embedding(request.id, request.text)
        return TextResponse(
            id=request.id,
            embeddings=[
                ChunkEmbedding(
                    chunk_number=c.chunk_number,
                    chunk=c.chunk,
                    embedding=c.embedding
                )
                for c in chunks
            ]
        )
    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embed/batch", response_model=BatchResponse)
async def embed_batch(request: BatchTextRequest):
    """Generate embeddings for multiple documents"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents array is required")

    if len(request.documents) == 0:
        raise HTTPException(status_code=400, detail="documents array cannot be empty")

    if len(request.documents) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch size {len(request.documents)} exceeds maximum of {settings.max_batch_size}"
        )

    # Validate documents
    for doc in request.documents:
        if doc.id is None:
            raise HTTPException(status_code=400, detail="each document must have an id")
        if not doc.text:
            raise HTTPException(status_code=400, detail=f"document {doc.id} is missing text")

    try:
        docs_data = [{"id": d.id, "text": d.text} for d in request.documents]
        results, timing = await embedding_service.generate_batch_embeddings(docs_data)

        return BatchResponse(
            results=[
                TextResponse(
                    id=r["id"],
                    embeddings=[
                        ChunkEmbedding(
                            chunk_number=e["chunk_number"],
                            chunk=e["chunk"],
                            embedding=e["embedding"]
                        )
                        for e in r["embeddings"]
                    ]
                )
                for r in results
            ],
            timing=TimingResponse(
                total_ms=timing.total_ms,
                chunking_ms=timing.chunking_ms,
                inference_ms=timing.inference_ms,
                postprocess_ms=timing.postprocess_ms
            )
        )
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
async def root():
    """Root endpoint with API info"""
    gpu_info = get_gpu_info()

    return {
        "name": "Inception GPU Inference Service",
        "version": "2.0.0-gpu",
        "provider": embedding_service.get_status()["provider"],
        "device": gpu_info["device_name"] if gpu_info else "CPU",
        "endpoints": [
            "GET  /health              - Health check",
            "GET  /status              - Detailed status",
            "POST /api/v1/embed/query  - Query embedding",
            "POST /api/v1/embed/text   - Document embedding",
            "POST /api/v1/embed/batch  - Batch embeddings",
        ],
    }


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info"
    )
