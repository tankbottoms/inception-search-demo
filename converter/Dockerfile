# Python ONNX Model Converter
# Converts HuggingFace models to ONNX format
# Supports embedding models and OCR models (HunyuanOCR)

FROM python:3.12-slim

WORKDIR /converter

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for better reliability
RUN pip config set global.timeout 300 && \
    pip config set global.retries 5

# Install dependencies in stages to avoid timeout
# Stage 1: Core ML dependencies (largest downloads)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Stage 2: HuggingFace ecosystem (from git for latest model support like HunyuanVL)
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git \
    sentence-transformers \
    tokenizers

# Stage 3: ONNX tools
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime \
    optimum

# Stage 4: Web server
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn

# Copy converter scripts
COPY convert.py server.py convert_hunyuan_ocr.py ./

VOLUME /models

ENV PORT=8010
ENV MODELS_DIR=/models
ENV AUTO_CONVERT=true
ENV HF_HOME=/models/.cache/huggingface

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Default: run HTTP server with auto-conversion
CMD ["python", "server.py"]
