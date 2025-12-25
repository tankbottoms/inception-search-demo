# Python ONNX Model Converter
# Converts HuggingFace models to ONNX format

FROM python:3.12-slim

WORKDIR /converter

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    optimum[exporters] \
    torch \
    transformers \
    sentence-transformers \
    fastapi \
    uvicorn

COPY convert.py server.py ./

VOLUME /models

ENV PORT=8010
ENV MODELS_DIR=/models

EXPOSE 8010

# Default: run HTTP server
CMD ["python", "server.py"]
