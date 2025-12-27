"""
Configuration for Python GPU Inference Service
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

@dataclass
class Settings:
    # Model settings
    model_name: str = "freelawproject--modernbert-embed-base_finetune_512-pytorch"
    model_cache_dir: Path = Path("../models")
    max_tokens: int = 512
    embedding_dim: int = 768

    # Text processing
    min_text_length: int = 1
    max_query_length: int = 1000
    max_text_length: int = 10_000_000
    overlap_ratio: float = 0.004

    # Batch processing
    max_batch_size: int = 100
    processing_batch_size: int = 32  # Higher for GPU

    # Hardware
    execution_provider: Literal["cuda", "cpu"] = "cuda"
    device_id: int = 0

    # Prefixes
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "

    # Server
    host: str = "0.0.0.0"
    port: int = 8006  # Different port from TypeScript service

    def __post_init__(self):
        # Override from environment
        self.model_name = os.getenv("TRANSFORMER_MODEL_NAME", self.model_name)
        self.max_tokens = int(os.getenv("MAX_TOKENS", self.max_tokens))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", self.max_batch_size))
        self.processing_batch_size = int(os.getenv("PROCESSING_BATCH_SIZE", self.processing_batch_size))

        if os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes"):
            self.execution_provider = "cpu"

        self.port = int(os.getenv("PORT", self.port))
        self.device_id = int(os.getenv("CUDA_DEVICE_ID", self.device_id))

        # Resolve model cache directory
        cache_dir = os.getenv("MODEL_CACHE_DIR", str(self.model_cache_dir))
        self.model_cache_dir = Path(cache_dir).resolve()

settings = Settings()
