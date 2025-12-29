#!/usr/bin/env python3
"""
ONNX Model Converter HTTP Server

Provides on-demand model conversion via REST API.
Auto-converts models from registry on startup.

Endpoints:
    GET  /           - Server info
    GET  /status     - Conversion status
    GET  /models     - List models and conversion status
    POST /convert    - Convert a single model
    POST /convert-all - Convert all models from registry
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from convert import convert_model, convert_from_registry, is_model_converted

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for tracking conversion status
conversion_status = {
    "state": "idle",  # idle, converting, complete, error
    "current_model": None,
    "results": None,
    "error": None
}


def run_startup_conversion():
    """Run model conversion from registry on startup"""
    global conversion_status

    models_dir = os.environ.get("MODELS_DIR", "/models")
    registry_path = os.path.join(models_dir, "registry.json")

    if not os.path.exists(registry_path):
        logger.warning(f"[STARTUP] Registry not found: {registry_path}")
        conversion_status["state"] = "idle"
        conversion_status["error"] = "Registry not found"
        return

    logger.info("[STARTUP] Starting auto-conversion from registry...")
    conversion_status["state"] = "converting"

    try:
        results = convert_from_registry(registry_path, models_dir)
        conversion_status["state"] = "complete"
        conversion_status["results"] = results
        logger.info("[STARTUP] Auto-conversion complete")
    except Exception as e:
        logger.error(f"[STARTUP] Auto-conversion failed: {e}")
        conversion_status["state"] = "error"
        conversion_status["error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup conversion before serving requests"""
    # Run conversion in a thread to not block startup
    logger.info("[SERVER] Starting converter service...")

    # Check if AUTO_CONVERT is enabled (default: True)
    auto_convert = os.environ.get("AUTO_CONVERT", "true").lower() == "true"

    if auto_convert:
        # Run in background thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_startup_conversion)
    else:
        logger.info("[SERVER] AUTO_CONVERT disabled, skipping startup conversion")

    yield

    logger.info("[SERVER] Shutting down converter service...")


app = FastAPI(
    title="Inception ONNX Converter",
    description="On-demand ONNX model conversion service with auto-convert on startup",
    version="2.0.0",
    lifespan=lifespan
)


class ConvertRequest(BaseModel):
    model_name: str
    model_type: str = "embedding"
    output_id: Optional[str] = None
    force: bool = False


class ConvertResponse(BaseModel):
    status: str
    output: Optional[str] = None
    error: Optional[str] = None


class ModelStatus(BaseModel):
    id: str
    name: str
    type: str
    enabled: bool
    converted: bool
    onnx_path: Optional[str] = None


@app.get("/")
def root():
    return {
        "name": "Inception ONNX Converter",
        "version": "2.0.0",
        "features": [
            "Auto-convert from registry on startup",
            "On-demand conversion via API",
            "Support for embedding and OCR models"
        ],
        "endpoints": [
            "GET /status",
            "GET /models",
            "POST /convert",
            "POST /convert-all"
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    models_dir = os.environ.get("MODELS_DIR", "/models")
    registry_path = os.path.join(models_dir, "registry.json")

    return {
        "status": conversion_status["state"],
        "current_model": conversion_status["current_model"],
        "results": conversion_status["results"],
        "error": conversion_status["error"],
        "models_dir": models_dir,
        "registry_exists": os.path.exists(registry_path)
    }


@app.get("/models")
def list_models():
    """List all models and their conversion status"""
    import json

    models_dir = os.environ.get("MODELS_DIR", "/models")
    registry_path = os.path.join(models_dir, "registry.json")

    if not os.path.exists(registry_path):
        raise HTTPException(status_code=404, detail="Registry not found")

    with open(registry_path) as f:
        registry = json.load(f)

    models_status = []

    for model in registry.get("models", []):
        model_id = model.get("id")
        model_name = model.get("name")
        model_type = model.get("type", "embedding")
        enabled = model.get("enabled", False)
        formats = model.get("formats", [])

        # Check if converted
        safe_name = model_name.replace("/", "--")
        converted = is_model_converted(safe_name, models_dir)

        # Find ONNX path if converted
        onnx_path = None
        if converted:
            model_path = Path(models_dir) / safe_name
            onnx_files = list(model_path.glob("*.onnx"))
            if onnx_files:
                onnx_path = str(onnx_files[0])

        models_status.append({
            "id": model_id,
            "name": model_name,
            "type": model_type,
            "enabled": enabled,
            "formats": formats,
            "converted": converted,
            "onnx_path": onnx_path
        })

    return {
        "models": models_status,
        "total": len(models_status),
        "converted": sum(1 for m in models_status if m["converted"]),
        "enabled": sum(1 for m in models_status if m["enabled"])
    }


@app.post("/convert", response_model=ConvertResponse)
def convert(request: ConvertRequest):
    """Convert a single model"""
    models_dir = os.environ.get("MODELS_DIR", "/models")
    output_id = request.output_id or request.model_name.replace("/", "--")
    output_dir = os.path.join(models_dir, output_id)

    # Check if already converted
    if not request.force and is_model_converted(output_id, models_dir):
        return ConvertResponse(
            status="skipped",
            output=output_dir
        )

    result = convert_model(request.model_name, output_dir, model_type=request.model_type)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])

    return ConvertResponse(
        status="success",
        output=output_dir
    )


@app.post("/convert-all")
async def convert_all(background_tasks: BackgroundTasks, force: bool = False):
    """Convert all models from registry (runs in background)"""
    global conversion_status

    if conversion_status["state"] == "converting":
        raise HTTPException(
            status_code=409,
            detail="Conversion already in progress"
        )

    models_dir = os.environ.get("MODELS_DIR", "/models")
    registry_path = os.path.join(models_dir, "registry.json")

    if not os.path.exists(registry_path):
        raise HTTPException(status_code=404, detail="Registry not found")

    def run_conversion():
        global conversion_status
        conversion_status["state"] = "converting"
        try:
            results = convert_from_registry(registry_path, models_dir, force=force)
            conversion_status["results"] = results
            conversion_status["state"] = "complete"
        except Exception as e:
            conversion_status["error"] = str(e)
            conversion_status["state"] = "error"

    background_tasks.add_task(run_conversion)

    return {
        "status": "started",
        "message": "Conversion started in background. Check /status for progress."
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8010))
    auto_convert = os.environ.get("AUTO_CONVERT", "true").lower() == "true"

    logger.info(f"[INFO] Starting converter server on port {port}")
    logger.info(f"[INFO] AUTO_CONVERT: {auto_convert}")

    uvicorn.run(app, host="0.0.0.0", port=port)
