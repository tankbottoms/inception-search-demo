#!/usr/bin/env python3
"""
ONNX Model Converter HTTP Server

Provides on-demand model conversion via REST API.

Endpoints:
    POST /convert - Convert a model to ONNX
    GET /status - Check converter status
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from convert import convert_model

app = FastAPI(
    title="Inception ONNX Converter",
    description="On-demand ONNX model conversion service",
    version="1.0.0"
)


class ConvertRequest(BaseModel):
    model_name: str
    output_id: str | None = None


class ConvertResponse(BaseModel):
    status: str
    output: str | None = None
    error: str | None = None


@app.get("/")
def root():
    return {
        "name": "Inception ONNX Converter",
        "version": "1.0.0",
        "endpoints": [
            "GET /status",
            "POST /convert"
        ]
    }


@app.get("/status")
def status():
    return {
        "status": "ready",
        "models_dir": os.environ.get("MODELS_DIR", "/models")
    }


@app.post("/convert", response_model=ConvertResponse)
def convert(request: ConvertRequest):
    models_dir = os.environ.get("MODELS_DIR", "/models")
    output_id = request.output_id or request.model_name.split("/")[-1]
    output_dir = os.path.join(models_dir, output_id)

    result = convert_model(request.model_name, output_dir)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])

    return ConvertResponse(
        status="success",
        output=output_dir
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8010))
    print(f"[INFO] Starting converter server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
