# Python GPU Inference Service
# Uses NVIDIA's vLLM container for proper CUDA support on Grace Hopper (GB10)
FROM nvcr.io/nvidia/vllm:25.12-py3

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    transformers \
    tokenizers \
    sentence-transformers \
    pydantic \
    python-dotenv \
    httpx \
    prometheus-client

# Copy application code
COPY src/ /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/models
ENV PORT=8006

# Expose port
EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

# Run the service
CMD ["python", "main.py"]
