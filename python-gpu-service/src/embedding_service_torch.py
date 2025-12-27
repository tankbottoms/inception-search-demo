"""
GPU-accelerated Embedding Service using PyTorch / Sentence Transformers
Falls back to ONNX Runtime if available
"""
import time
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch and sentence-transformers
try:
    import torch
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.warning("PyTorch/sentence-transformers not available, falling back to ONNX")

# Try ONNX Runtime as fallback
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from config import settings


@dataclass
class TimingInfo:
    total_ms: float
    chunking_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    postprocess_ms: Optional[float] = None


@dataclass
class ChunkEmbedding:
    chunk_number: int
    chunk: str
    embedding: List[float]


class EmbeddingService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.ort_session = None
        self.initialized = False
        self.model_id: Optional[str] = None
        self.provider: str = "unknown"
        self.backend: str = "none"

    def _get_model_path(self) -> Path:
        """Get path to model directory"""
        model_name = settings.model_name.replace("/", "--")
        return settings.model_cache_dir / model_name

    async def initialize(self, model_id: str = "modernbert-embed") -> None:
        """Initialize the embedding service with GPU support"""
        if self.initialized and self.model_id == model_id:
            logger.debug("Embedding service already initialized")
            return

        start_time = time.perf_counter()
        model_path = self._get_model_path()

        # Try PyTorch/sentence-transformers first (for GPU support)
        if TORCH_AVAILABLE:
            try:
                device = "cuda" if CUDA_AVAILABLE else "cpu"
                logger.info(f"Loading model with sentence-transformers on {device}")

                # Load from local path or HuggingFace
                if model_path.exists():
                    self.model = SentenceTransformer(str(model_path), device=device)
                else:
                    self.model = SentenceTransformer(settings.model_name, device=device)
                    # Save locally for future use
                    self.model.save(str(model_path))

                self.backend = "torch"
                self.provider = f"CUDA ({torch.cuda.get_device_name(0)})" if CUDA_AVAILABLE else "CPU"
                self.model_id = model_id
                self.initialized = True

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"Embedding service initialized in {elapsed:.2f}ms using {self.backend}")
                logger.info(f"Provider: {self.provider}")
                return

            except Exception as e:
                logger.warning(f"Failed to load with sentence-transformers: {e}")

        # Fall back to ONNX Runtime
        if ONNX_AVAILABLE:
            try:
                logger.info("Falling back to ONNX Runtime")
                onnx_files = list(model_path.glob("*.onnx"))
                if not onnx_files:
                    raise FileNotFoundError(f"No ONNX model in {model_path}")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

                # Create ONNX session
                providers = ort.get_available_providers()
                self.ort_session = ort.InferenceSession(str(onnx_files[0]), providers=providers)

                self.backend = "onnx"
                self.provider = self.ort_session.get_providers()[0]
                self.model_id = model_id
                self.initialized = True

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"Embedding service initialized in {elapsed:.2f}ms using {self.backend}")
                logger.info(f"Provider: {self.provider}")
                return

            except Exception as e:
                logger.error(f"ONNX Runtime initialization failed: {e}")

        raise RuntimeError("No suitable backend available (tried PyTorch and ONNX)")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding"""
        if not text:
            return ""

        # Normalize unicode
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')

        # Normalize whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('\t', ' ')
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'ltd',
            'corp', 'co', 'no', 'vol', 'rev', 'st', 'ave', 'blvd', 'rd', 'ct',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        }

        normalized = re.sub(r'\s+', ' ', text).strip()
        if not normalized:
            return []

        parts = re.split(r'(?<=[.!?])\s+', normalized)
        sentences = []
        current = ""

        for part in parts:
            if not part.strip():
                continue
            last_word = current.strip().split()[-1].lower().rstrip('.') if current.strip() else ""
            if last_word in abbreviations and current:
                current += " " + part
            elif current:
                sentences.append(current.strip())
                current = part
            else:
                current = part

        if current.strip():
            sentences.append(current.strip())

        return sentences

    def _split_into_chunks(self, text: str, prefix: str) -> List[str]:
        """Split text into chunks respecting token limits"""
        processed = self._preprocess_text(text)
        if not processed:
            return []

        sentences = self._sentence_tokenize(processed)
        if not sentences:
            return []

        max_tokens = settings.max_tokens
        overlap_sentences = max(1, int(max_tokens * settings.overlap_ratio))

        # Simplified chunking - split by character count as approximation
        # Each token is roughly 4 characters
        max_chars = max_tokens * 4

        chunks = []
        current_chunk: List[str] = []
        current_len = len(prefix)

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_len + sentence_len > max_chars:
                if current_chunk:
                    chunks.append(prefix + " ".join(current_chunk))
                overlap = current_chunk[-overlap_sentences:] if len(current_chunk) >= overlap_sentences else []
                current_chunk = overlap + [sentence]
                current_len = len(prefix) + sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_len += sentence_len

        if current_chunk:
            chunks.append(prefix + " ".join(current_chunk))

        return chunks

    async def _run_inference_torch(self, texts: List[str]) -> np.ndarray:
        """Run inference using PyTorch/sentence-transformers"""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=settings.processing_batch_size
        )
        return embeddings

    async def _run_inference_onnx(self, texts: List[str]) -> np.ndarray:
        """Run inference using ONNX Runtime"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_tokens,
            return_tensors="np"
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}

        input_names = [inp.name for inp in self.ort_session.get_inputs()]
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.ort_session.run(None, feeds)
        hidden_states = outputs[0]

        # Mean pooling
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_hidden = np.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.maximum(sum_mask, 1e-9)
        embeddings = sum_hidden / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)

        return embeddings

    async def _run_inference(self, texts: List[str]) -> np.ndarray:
        """Run inference using available backend"""
        if self.backend == "torch":
            return await self._run_inference_torch(texts)
        elif self.backend == "onnx":
            return await self._run_inference_onnx(texts)
        else:
            raise RuntimeError("No backend initialized")

    async def generate_query_embedding(self, text: str) -> Tuple[List[float], TimingInfo]:
        """Generate embedding for a query"""
        start = time.perf_counter()

        if not self.initialized:
            await self.initialize()

        processed = self._preprocess_text(text)
        query_text = settings.query_prefix + processed

        preprocess_time = time.perf_counter()
        embeddings = await self._run_inference([query_text])
        inference_time = time.perf_counter()

        total_ms = (inference_time - start) * 1000
        inference_ms = (inference_time - preprocess_time) * 1000

        return embeddings[0].tolist(), TimingInfo(total_ms=total_ms, inference_ms=inference_ms)

    async def generate_text_embedding(self, doc_id: int, text: str) -> Tuple[List[ChunkEmbedding], TimingInfo]:
        """Generate embeddings for a document with chunking"""
        start = time.perf_counter()

        if not self.initialized:
            await self.initialize()

        chunks = self._split_into_chunks(text, settings.document_prefix)
        if not chunks:
            raise ValueError("No content to embed after processing")

        chunking_time = time.perf_counter()

        # Process in batches
        all_embeddings = []
        batch_size = settings.processing_batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = await self._run_inference(batch)
            all_embeddings.extend(embeddings)

        inference_time = time.perf_counter()

        results = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            chunk_text = chunk.replace(settings.document_prefix, "", 1)
            results.append(ChunkEmbedding(
                chunk_number=i + 1,
                chunk=chunk_text,
                embedding=embedding.tolist()
            ))

        postprocess_time = time.perf_counter()

        return results, TimingInfo(
            total_ms=(postprocess_time - start) * 1000,
            chunking_ms=(chunking_time - start) * 1000,
            inference_ms=(inference_time - chunking_time) * 1000,
            postprocess_ms=(postprocess_time - inference_time) * 1000
        )

    async def generate_batch_embeddings(
        self,
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], TimingInfo]:
        """Generate embeddings for multiple documents"""
        start = time.perf_counter()

        if not self.initialized:
            await self.initialize()

        all_chunks = []
        chunk_meta = []

        for doc in documents:
            doc_id = doc["id"]
            chunks = self._split_into_chunks(doc["text"], settings.document_prefix)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_meta.append({"doc_id": doc_id, "doc_idx": documents.index(doc)})

        chunking_time = time.perf_counter()
        logger.info(f"Processing batch: {len(documents)} docs, {len(all_chunks)} chunks")

        all_embeddings = []
        batch_size = settings.processing_batch_size

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = await self._run_inference(batch)
            all_embeddings.extend(embeddings)

        inference_time = time.perf_counter()

        results = {doc["id"]: {"id": doc["id"], "embeddings": []} for doc in documents}

        for i, (chunk, embedding, meta) in enumerate(zip(all_chunks, all_embeddings, chunk_meta)):
            chunk_text = chunk.replace(settings.document_prefix, "", 1)
            doc_result = results[meta["doc_id"]]
            doc_result["embeddings"].append({
                "chunk_number": len(doc_result["embeddings"]) + 1,
                "chunk": chunk_text,
                "embedding": embedding.tolist()
            })

        postprocess_time = time.perf_counter()

        return list(results.values()), TimingInfo(
            total_ms=(postprocess_time - start) * 1000,
            chunking_ms=(chunking_time - start) * 1000,
            inference_ms=(inference_time - chunking_time) * 1000,
            postprocess_ms=(postprocess_time - inference_time) * 1000
        )

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "initialized": self.initialized,
            "model_id": self.model_id,
            "provider": self.provider,
            "backend": self.backend,
            "embedding_dim": settings.embedding_dim,
            "cuda_available": CUDA_AVAILABLE if TORCH_AVAILABLE else False,
        }


# Global instance
embedding_service = EmbeddingService()
